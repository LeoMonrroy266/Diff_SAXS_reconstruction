# coding:utf-8
import numpy as np
import map2iq_shape as map2iq
import auto_encoder_t_py3
import time
import multiprocessing
import threading
import region_search
import os
import result2pdb
import argparse
import processSaxs as ps
from functools import partial
import tensorflow as tf
import os
import logging
import absl.logging
import align2PDB as align2pdb
# Silence prints in terminal to see program output when running/debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
tf.get_logger().setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)

GPU_NUM = 1
BATCH_SIZE = 10

np.set_printoptions(precision=10)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',     help='path to Keras model/weights used by auto_encoder_t_py3', type=str, required=True)
parser.add_argument('--iq_path',        help='path of iq_file', type=str, required=True)
parser.add_argument('--rmax',           help='radius of the protein', default=0, type=float)
parser.add_argument('--output_folder',  help='path of output file', type=str, required=True)
parser.add_argument('--target_pdb',     help='path of target pdb file', default='None', type=str)
parser.add_argument('--rmax_start',     help='start range of rmax', default=10, type=float)
parser.add_argument('--rmax_end',       help='end range of rmax', default=300, type=float)
parser.add_argument('--max_iter',       help='maximum number of iteration', default=80, type=int)
args = parser.parse_args()

weights_path = args.model_path  # Keras weights/model file for auto_encoder_t_py3

# group_init_parameter is well-trained model's distribution of latent vector, used to initialize gene group.
group_init_parameter = np.loadtxt(os.path.join(os.path.dirname(weights_path), 'genegroup_init_parameter_2.txt'), delimiter=' ')
AUTO, ENCODER, DECODER = auto_encoder_t_py3.build_autoencoder()
auto_encoder_t_py3._load_weights(AUTO, weights_path)        # sets weights for encoder+decoder


class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
        self.result = None

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


class Evolution:
    def __init__(self, output_folder, mode, rmax_start, rmax_end, process_result, max_iter=80):
        # mode has 'withrmax' and 'withoutrmax' (know the size or not)
        self.mode = mode
        self.rmax_start = rmax_start
        self.rmax_end = rmax_end
        self.output_folder = output_folder
        self.iteration_step = 0
        self.counter = 0
        self.max_iter = max_iter
        self.process_result = process_result

        # length of latent vector
        self.gene_length = 200

        # numbers of two-point crossing one time.
        self.exchange_gene_num = 2

        # initial group_num
        self.group_num = 300
        self.inheritance_num = 300

        # every iteration step, keep top 20 samples unchanged
        self.remain_best_num = 20

        # used for averaging when getting the final result
        self.statistics_num = 20

        self.compute_score_withoutrmax = map2iq.run_withoutrmax
        self.compute_score_withrmax = map2iq.run_withrmax

        self.group = self.generate_original_group(self.group_num)
        self.group_score = self.compute_group_score(self.group)
        self.group, self.group_score = self.rank_group(self.group, self.group_score)

        if self.mode == 'withoutrmax':
            self.topXnum = int(100)
            self.topXrmax = np.copy(self.group[:self.topXnum, -1])

        self.best_so_far = np.copy(self.group[:self.remain_best_num])
        self.best_so_far_score = np.copy(self.group_score[:self.remain_best_num])
        self.score_mat = np.copy(self.group_score[:self.statistics_num]).reshape((1, self.statistics_num))
        if self.mode == 'withoutrmax':
            self.gene_data = np.copy(self.group[:self.statistics_num]).reshape((1, self.statistics_num, 201))
        else:
            self.gene_data = np.copy(self.group[:self.statistics_num]).reshape((1, self.statistics_num, 200))

        print('original input , top5:', self.group_score[:5])
        print('best_so_far, top5:', self.best_so_far_score[:5])
        print('mean_score is:', np.mean(self.group_score))
        print('initialized')

    # ---------------- TF2 helpers (encode/decode) ----------------
    def _encode_voxels(self, voxel):
        z = ENCODER.predict(voxel, batch_size=BATCH_SIZE, verbose=0)
        return z

    def _decode_latent(self, z_vectors):
        v = DECODER.predict(z_vectors, batch_size=BATCH_SIZE, verbose=0)
        return v


    # every iteration, ensure all samples have only one connected area.
    def region_process(self, cube_group, gpu_id=None):
        num = cube_group.shape[0]
        z_group = []
        real_data_group = []

        # preallocate padded array
        in_ = np.zeros((1, 32, 32, 32, 1), dtype=np.float32)

        for ii in range(num):
            out = cube_group[ii]  # (31,31,31)

            while True:
                in_[0, :31, :31, :31, 0] = out  # no astype needed if already float32

                # encode → z, decode z
                z_ = self._encode_voxels(in_)
                out_ = self._decode_latent(z_)

                z_ = z_.reshape(self.gene_length)
                real_data = (out_[0, :31, :31, :31, 0] > 0.1).astype(np.int32)

                # keep largest connected region
                out_mask, region_num = region_search.find_largest_connected_region(real_data)

                if region_num <= 1:
                    break
                out = out_mask

            z_group.append(z_.reshape(1, self.gene_length))
            real_data_group.append(real_data.reshape(1, 31, 31, 31))

        z_group = np.concatenate(z_group, axis=0)
        real_data_group = np.concatenate(real_data_group, axis=0)
        return z_group, real_data_group

    # use trained autoencoder model to get 3D structure from latent vector.
    def run_decode(self, data, _ii_unused):
        # data: (N, gene_length)
        n = data.shape[0]
        # batch through the decoder
        rec = self._decode_latent(data.astype(np.float32))
        rec = (rec > 0.1).astype(np.int32)
        rec = rec[:, :31, :31, :31, 0].reshape((-1, 31, 31, 31))
        return rec

    # decode a whole group (kept for API compatibility, now single-threaded batching)
    def multi_thread_decode_group(self, group):
        # group: (N, 200)
        source_num = group.shape[0]
        # pad to a multiple of BATCH_SIZE for convenience
        data_size = (source_num + BATCH_SIZE - 1) // BATCH_SIZE
        if source_num % BATCH_SIZE != 0:
            group_copy = np.copy(group)
            pad = data_size * BATCH_SIZE - source_num
            group_copy = np.vstack([group_copy, np.zeros((pad, group.shape[1]), dtype=group.dtype)])
            group = group_copy

        # run in simple batches (Keras uses TF2 eager/graph internally)
        batches = []
        for i in range(data_size):
            start = i * BATCH_SIZE
            stop = (i + 1) * BATCH_SIZE
            sub = self.run_decode(group[start:stop], 0)
            batches.append(sub)
        real_data_group = np.concatenate(batches, axis=0)
        return real_data_group[:source_num, :, :, :]

    # get scores of all the group based on fitness Function.
    def compute_group_score(self, group):
        decodetime1 = time.time()
        real_data_group = self.multi_thread_decode_group(group[:, :self.gene_length])
        decodetime2 = time.time()
        logfile.write('decode_time:%d\n' % (decodetime2 - decodetime1))

        num = group.shape[0]
        if self.mode == 'withoutrmax':
            group_rmax = np.copy(group[:, -1]).reshape(-1, 1)

        t1 = time.time()

        result = np.array([region_search.find_largest_connected_region(x) for x in real_data_group])



        real_data_group = result
        result = np.array([np.pad(r, pad_width=((0, 1), (0, 1), (0, 1)), mode='constant') for r in result])

        self.group[:,:self.gene_length] = self._encode_voxels(result)

        #logfile.write('find_region_time:%d\n' % (t2 - t1))

        compute_score_time1 = time.time()

        if self.mode == 'withoutrmax':
            compute_score_input_voxel = real_data_group.reshape(real_data_group.shape[0], -1)
            compute_score_input = np.concatenate([compute_score_input_voxel, self.group[:, -1].reshape(-1, 1)], axis=1)
            pool = multiprocessing.Pool(processes=20)
            func = partial(self.compute_score_withoutrmax, iq_file= self.process_result)
            result = np.array(pool.map(func,compute_score_input))

            pool.close()
            pool.join()
            group_score = np.copy(result[:, 0])
            self.group[:, -1] = result[:, 1]

        elif self.mode == 'withrmax':
            pool = multiprocessing.Pool(processes=20)
            func = partial(self.compute_score_withrmax,  iq_file=self.process_result)
            result = np.array(pool.map(func, real_data_group))
            pool.close()
            pool.join()
            group_score = np.array(result)

        compute_score_time2 = time.time()
        logfile.write('compute_score_time:%d\n' % (compute_score_time2 - compute_score_time1))

        return group_score

    # rank whole group based on their scores.
    def rank_group(self, group, group_score):
        index = np.argsort(group_score)
        group = group[index]
        group_score = group_score[index]
        return group, group_score

    # two-point crossing
    def exchange_gene(self, selective_gene):
        np.random.shuffle(selective_gene)
        for ii in range(0, self.inheritance_num - self.remain_best_num, 2):
            cross_point = np.random.randint(0, self.gene_length, size=(2 * self.exchange_gene_num))
            cross_point = np.sort(cross_point)
            for jj in range(self.exchange_gene_num):
                random_data = np.random.uniform(low=0, high=1)
                if random_data < 0.8:
                    a = cross_point[jj * 2]
                    b = cross_point[jj * 2 + 1]
                    temp = np.copy(selective_gene[ii, a:b])
                    selective_gene[ii, a:b] = selective_gene[ii + 1, a:b]
                    selective_gene[ii + 1, a:b] = np.copy(temp)

    # mutation operator
    def gene_variation(self, selective_gene):
        if self.mode == 'withoutrmax':
            average_rmax = np.mean(self.topXrmax)
            std_rmax = np.std(self.topXrmax)
        for ii in range(self.inheritance_num - self.remain_best_num):
            random_data = np.random.uniform(low=0, high=1, size=(self.gene_length + 1))
            for jj in range(self.gene_length):
                if random_data[jj] < 0.05:
                    gene_point = np.random.normal(group_init_parameter[jj, 0], group_init_parameter[jj, 1], size=1)
                    gene_point = abs(gene_point)
                    selective_gene[ii, jj] = gene_point

            if self.mode == 'withoutrmax':
                if random_data[-1] < 0.5:
                    random_num = np.random.uniform(low=0, high=1, size=1)
                    if random_num < 0.5:
                        rmax_variation = np.random.randint(self.rmax_start, self.rmax_end)
                        selective_gene[ii, -1] = rmax_variation
                    else:
                        rmax_variation = np.random.normal(average_rmax, std_rmax, size=1)
                        while rmax_variation <= 10:
                            rmax_variation = np.random.normal(average_rmax, std_rmax, size=1)
                        selective_gene[ii, -1] = rmax_variation

    # selection operator
    def select_group(self):
        if self.mode == 'withoutrmax':
            selected_group = np.zeros(shape=(self.inheritance_num - self.remain_best_num, self.gene_length + 1))
        elif self.mode == 'withrmax':
            selected_group = np.zeros(shape=(self.inheritance_num - self.remain_best_num, self.gene_length))

        selected_group_score = np.zeros(shape=(self.inheritance_num - self.remain_best_num))
        for ii in range(self.inheritance_num - self.remain_best_num):
            a = np.random.randint(0, self.group_num)
            b = np.random.randint(0, self.group_num)
            random_data = np.random.uniform(low=0, high=1)
            if random_data > 0.1:
                if a < b:
                    selected_group[ii] = np.copy(self.group[a])
                    selected_group_score[ii] = np.copy(self.group_score[a])
                else:
                    selected_group[ii] = np.copy(self.group[b])
                    selected_group_score[ii] = np.copy(self.group_score[b])
            else:
                if a < b:
                    selected_group[ii] = np.copy(self.group[b])
                    selected_group_score[ii] = np.copy(self.group_score[b])
                else:
                    selected_group[ii] = np.copy(self.group[a])
                    selected_group_score[ii] = np.copy(self.group_score[a])

        self.group = selected_group
        self.group_score = selected_group_score

    def inheritance(self):
        self.select_group()
        self.exchange_gene(self.group)
        self.gene_variation(self.group)
        if self.group.shape[0] != self.inheritance_num - self.remain_best_num:
            raise Exception('bad')

        self.group = np.concatenate((self.group, self.best_so_far), axis=0)
        t1 = time.time()
        self.group_score = self.compute_group_score(self.group)
        t2 = time.time()
        logfile.write('compute_group_score cost:%d\n' % (t2 - t1))

        self.group, self.group_score = self.rank_group(self.group, self.group_score)
        if self.mode == 'withoutrmax':
            self.topXrmax = np.copy(self.group[:self.topXnum, -1])
            self.gene_data = np.concatenate((self.gene_data, self.group[:self.statistics_num].reshape((1, self.statistics_num, 201))), axis=0)
        elif self.mode == 'withrmax':
            self.gene_data = np.concatenate((self.gene_data, self.group[:self.statistics_num].reshape((1, self.statistics_num, 200))), axis=0)
        self.score_mat = np.concatenate((self.score_mat, self.group_score[:self.statistics_num].reshape((1, self.statistics_num))), axis=0)
        self.best_so_far = np.copy(self.group[:self.remain_best_num])
        self.best_so_far_score = np.copy(self.group_score[:self.remain_best_num])
        self.group = np.copy(self.group[:self.group_num])
        self.group_score = np.copy(self.group_score[:self.group_num])

    # If best sample remains unchanged 15 times, reduce the size of the group.
    # Termination: best unchanged 15 times when group size is 100, or max_iter exceeded.
    def evolution_iteration(self):
        while True:
            t1 = time.time()
            self.inheritance()
            self.iteration_step = self.iteration_step + 1
            t2 = time.time()
            print('iteration_step:', self.iteration_step, 'top5:', self.group_score[:5],
                  '\nmean_score is:%.2f' % np.mean(self.score_mat[-1]), self.group_num)
            logfile.write('iteration_step_%d' % self.iteration_step)
            logfile.write(' cost:%d \n\n' % (t2 - t1))

            if self.score_mat[-1, 0] < self.score_mat[-2, 0]:
                self.counter = 0
            else:
                self.counter = self.counter + 1
                if self.counter > 10 or self.iteration_step > self.max_iter:
                    self.group_num = self.group_num - 100
                    self.inheritance_num = self.inheritance_num - 100
                    self.counter = 0
                    if self.group_num < 100 or self.iteration_step > self.max_iter:
                        np.savetxt('%s/score_mat.txt' % self.output_folder, self.score_mat, fmt='%.3f')
                        result_sample = self.multi_thread_decode_group(self.group[:self.statistics_num, :self.gene_length])
                        t3 = time.time()
                        if self.mode == 'withoutrmax':
                            gene = self.gene_data.reshape((-1, self.gene_length + 1))
                            voxel_group = self.multi_thread_decode_group(gene[:, :-1])
                            voxel_group = voxel_group.reshape((-1, self.statistics_num, 31, 31, 31))
                            t4 = time.time()
                            logfile.write('\nvoxel_group cost:%d\n' % (t4 - t3))
                            np.savetxt('%s/bestgene.txt' % output_folder, self.group[0], fmt='%.3f')
                            return result_sample, voxel_group, self.group[:self.statistics_num, -1], gene[:, -1].reshape((-1, self.statistics_num))
                        else:
                            gene = self.gene_data.reshape((-1, self.gene_length))
                            voxel_group = self.multi_thread_decode_group(gene)
                            voxel_group = voxel_group.reshape((-1, self.statistics_num, 31, 31, 31))
                            voxel_group = [region_search.find_largest_connected_region(v)[0] for v in voxel_group]
                            t4 = time.time()
                            logfile.write('\nvoxel_group cost:%d\n' % (t4 - t3))
                            np.savetxt('%s/bestgene.txt' % output_folder, self.group[0], fmt='%.3f')
                            return result_sample, voxel_group

    # group initialization
    def generate_original_group(self, num):
        original_group = np.zeros(shape=(num, 200))
        for ii in range(200):
            original_group[:, ii] = np.random.normal(group_init_parameter[ii, 0], group_init_parameter[ii, 1], size=num)
        original_group = abs(original_group)
        if self.mode == 'withoutrmax':
            original_rmax = np.random.randint(self.rmax_start, self.rmax_end, (num, 1)).astype(float)
            original_group = np.concatenate([original_group, original_rmax], axis=1)
        return original_group


if __name__ == '__main__':
    iq_path = args.iq_path
    rmax = args.rmax
    real_rmax = args.rmax
    output_folder = args.output_folder
    target_pdb = args.target_pdb
    rmax_start = args.rmax_start
    rmax_end = args.rmax_end + 1
    max_iter = args.max_iter

    os.makedirs(output_folder, exist_ok=True)

    estimate_rmax = None
    process_result = ps.process(iq_path)
    if len(process_result) == 2:
        estimate_rmax = process_result[1]
        estimate_rmax = None
    saxs_data = process_result[0]
    processed_saxs_path = os.path.join(output_folder, 'processed_saxs.iq')
    np.savetxt(processed_saxs_path, saxs_data, fmt='%.3f')
    saxs_data = map2iq.read_iq_ascii(processed_saxs_path)
    if rmax == 0 and (estimate_rmax is not None):
        rmax = float(estimate_rmax)

    BATCH_SIZE = 10
    #map2iq.output_folder = output_folder

    logfile = open('%s/log.txt' % output_folder, 'a')
    t1 = time.time()
    print('generate computing graph (TF2/Keras models are built on demand in auto_encoder_t_py3)')

    evolution_mode = 'withoutrmax' if rmax == 0 else 'withrmax'
    print('weights_path:', weights_path)

    # No tf.Session / Saver in TF2 — everything uses Keras predict()
    genetic_object = Evolution(output_folder, evolution_mode, rmax_start, rmax_end, max_iter=max_iter,process_result=saxs_data)
    if evolution_mode == 'withoutrmax':
        result_sample, voxel_group, result_sample_rmax, voxel_group_rmax = genetic_object.evolution_iteration()
        print("#####result:",result_sample.shape)
        t2 = time.time()
        print('result_sample_rmax:', result_sample_rmax)
        rmax = np.mean(result_sample_rmax)
        print('rmax_find', rmax)
        np.savetxt('%s/rmax_find_log.txt' % output_folder, voxel_group_rmax, fmt='%d')
    else:
        rmax = args.rmax
        print('rmax_real:', rmax)
        result_sample, voxel_group = genetic_object.evolution_iteration()
        t2 = time.time()

    if target_pdb == 'None':
        for n, result in enumerate(result_sample[:5]):
            result2pdb.write_single_pdb(result, output_folder, f'result_{n}.pdb', rmax=rmax)
        t3 = time.time()
    else:
        for n, result in enumerate(result_sample[:5]):
            result2pdb.write_single_pdb(result, output_folder, f'result_{n}.pdb', rmax=rmax)
            file = os.path.join(output_folder, f'result_{n}.pdb')
            save_file = os.path.join(output_folder, f'result_{n}_aligned.pdb')
            cc_global, cc_masked_hard, cc_masked_soft, cc_rscc, phi = align2pdb.main(file, target_pdb, save_file)
            #logfile.write(f"Overall PCC for model_{n} against ref: {cc_global} \n")
            #logfile.write(f"Masked_hard PCC for model_{n} against ref: {cc_masked_hard} \n")
            #logfile.write(f"Masked_soft PCC for model_{n} against ref: {cc_masked_soft} \n")
            logfile.write(f"Real space CC for model_{n} against ref: {cc_rscc} \n")
            logfile.write(f"Phi_max for model_{n} against ref: {phi} \n")
        t3 = time.time()


    t4 = time.time()
    print('total_time:', (t4 - t1))
    logfile.write('evolution time: %d\n' % (t2 - t1))
    logfile.write('write2pdb time: %d\n' % (t3 - t2))
    logfile.write('cal_cc time: %d\n' % (t4 - t3))
    logfile.write('total time: %d\n' % (t4 - t1))
    logfile.close()
