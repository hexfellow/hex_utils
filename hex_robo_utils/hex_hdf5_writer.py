#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-09-19
################################################################

import time
import threading
import queue
import h5py, hdf5plugin
import numpy as np


class HexHdf5Writer:

    def __init__(self, file_path: str):
        hdf5plugin.register()
        self.__file_path = file_path
        self.__hdf5_file = h5py.File(file_path, "w", libver='latest')
        self.__group_dict = {}
        self.__dataset_dict = {}

        self.__queue = queue.Queue(maxsize=1024)
        self.__stop_event = threading.Event()
        self.__batch_size = 32
        self.__writer_cnt = 0
        self.__writer_thread = None
        self.__writer_exc = None

    def __del__(self):
        self.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        if self.__writer_thread and self.__writer_thread.is_alive():
            return
        self.__stop_event.clear()
        self.__writer_thread = threading.Thread(
            target=self.__writer_loop,
            daemon=True,
        )
        self.__writer_thread.start()

    def stop(self):
        self.__stop_event.set()
        self.__writer_thread.join(timeout=2)
        if self.__hdf5_file is not None:
            try:
                self.__hdf5_file.flush()
                self.__hdf5_file.close()
            except Exception:
                pass
            self.__hdf5_file = None
        if self.__writer_exc:
            raise self.__writer_exc

    def get_shape(self, group_name: str, dataset_name: str = "data"):
        dataset = self.__get_dataset_handle(group_name, dataset_name)
        return dataset.shape

    def get_dtype(self, group_name: str, dataset_name: str = "data"):
        dataset = self.__get_dataset_handle(group_name, dataset_name)
        return dataset.dtype

    def __get_dataset_handle(
        self,
        group_name: str,
        dataset_name: str = "data",
    ):
        dataset_key = f"{group_name}/{dataset_name}"
        if dataset_key not in self.__dataset_dict:
            if group_name not in self.__hdf5_file:
                raise KeyError(f"Group '{group_name}' not found in HDF5 file")
            if dataset_name not in self.__hdf5_file[group_name]:
                raise KeyError(
                    f"Dataset '{dataset_name}' not found in group '{group_name}'"
                )
            self.__dataset_dict[dataset_key] = self.__hdf5_file[group_name][
                dataset_name]
        return self.__dataset_dict[dataset_key]

    def summary(self):
        if self.__hdf5_file is None:
            print("HDF5 file is closed. Cannot get summary.")
            return

        print("#" * 50)
        print(f"HDF5 File: {self.__file_path}")
        print("#" * 50)

        for group_name in self.__hdf5_file.keys():
            print("-" * 30)
            print(f"Group: {group_name}")
            print("-" * 30)
            print(
                f"get_delta: {(self.__hdf5_file[group_name]['get_ts'][-1] - self.__hdf5_file[group_name]['get_ts'][0]) / 1e9}s"
            )
            print(
                f"sen_delta: {(self.__hdf5_file[group_name]['sen_ts'][-1] - self.__hdf5_file[group_name]['sen_ts'][0]) / 1e9}s"
            )

            dtype = self.get_dtype(group_name)
            shape = self.get_shape(group_name)
            print(f"  Dataset: {group_name}")
            print(f"    Dtype: {dtype}")
            print(f"    Shape: {shape}")

        print("#" * 50)

    def __writer_loop(self):
        try:
            batch = []
            while not (self.__stop_event.is_set() and self.__queue.empty()):
                try:
                    item = self.__queue.get(timeout=0.1)
                    batch.append(item)
                    if len(batch) >= self.__batch_size:
                        self.__flush_batch(batch)
                        batch.clear()
                    continue
                except queue.Empty:
                    if batch:
                        self.__flush_batch(batch)
                        batch.clear()
                    continue
            if batch:
                self.__flush_batch(batch)
        except Exception as e:
            self.__writer_exc = e
            raise

    def __flush_batch(self, batch):
        buckets = {}
        for group, data, gts, sts in batch:
            buckets.setdefault(group, []).append((data, gts, sts))

        for group, items in buckets.items():
            dataset_key = f"{group}/data"
            get_ts_key = f"{group}/get_ts"
            sen_ts_key = f"{group}/sen_ts"
            ds = self.__dataset_dict[dataset_key]
            d_get = self.__dataset_dict[get_ts_key]
            d_sen = self.__dataset_dict[sen_ts_key]

            # 将 items 合并成 numpy arrays
            data_arr = np.stack([it[0] for it in items], axis=0)
            gts_arr = np.stack([it[1] for it in items], axis=0).reshape(-1, 1)
            sts_arr = np.stack([it[2] for it in items], axis=0).reshape(-1, 1)

            n_old = ds.shape[0]
            n_new = n_old + data_arr.shape[0]
            # 一次 resize（比逐帧 resize 快得多）
            ds.resize((n_new, *ds.shape[1:]))
            d_get.resize((n_new, 1))
            d_sen.resize((n_new, 1))

            ds[n_old:n_new, ...] = data_arr
            d_get[n_old:n_new, :] = gts_arr
            d_sen[n_old:n_new, :] = sts_arr

        try:
            self.__hdf5_file.flush()
        except Exception:
            pass

        batch_count = sum(len(items) for items in buckets.values())
        self.__writer_cnt += batch_count
        if self.__writer_cnt % (self.__batch_size * 32) == 0:
            print("#" * 50)
            for group_name in self.__hdf5_file.keys():
                print(f"{group_name} len:{self.get_shape(group_name)[0]}")

    def create_dataset(
            self,
            group_name: str,
            shape: tuple,
            dtype: np.dtype,
            chunk_num: int,
            compression=hdf5plugin.Bitshuffle(nelems=0, cname='lz4'),
    ):
        if group_name not in self.__group_dict:
            self.__group_dict[group_name] = self.__hdf5_file.create_group(
                group_name)

        dataset = self.__group_dict[group_name].create_dataset(
            "data",
            shape=(0, *shape),
            maxshape=(None, *shape),
            dtype=dtype,
            chunks=(chunk_num, *shape),
            compression=compression,
        )
        get_ts_set = self.__group_dict[group_name].create_dataset(
            "get_ts",
            shape=(0, 1),
            maxshape=(None, 1),
            dtype=np.int64,
            chunks=(chunk_num, 1),
        )
        sen_ts_set = self.__group_dict[group_name].create_dataset(
            "sen_ts",
            shape=(0, 1),
            maxshape=(None, 1),
            dtype=np.int64,
            chunks=(chunk_num, 1),
        )

        # Store dataset reference for easy access
        self.__dataset_dict[f"{group_name}/data"] = dataset
        self.__dataset_dict[f"{group_name}/get_ts"] = get_ts_set
        self.__dataset_dict[f"{group_name}/sen_ts"] = sen_ts_set

    def append_data(
        self,
        group_name: str,
        data: np.ndarray,
        get_ts: np.ndarray,
        sen_ts: np.ndarray,
        block: bool = True,
        timeout: float = None,
    ):
        item = (
            group_name,
            data,
            get_ts,
            sen_ts,
        )
        self.__queue.put(item, block=block, timeout=timeout)

    def append_batch_data(
        self,
        group_name: str,
        data: np.ndarray,
        get_ts: np.ndarray,
        sen_ts: np.ndarray,
        block: bool = True,
        timeout: float = None,
    ):
        for i in range(data.shape[0]):
            self.append_data(
                group_name,
                data[i],
                get_ts[i],
                sen_ts[i],
                block=block,
                timeout=timeout,
            )

    def now_ns(self):
        return np.array([time.time_ns()])

    def hex_ts_to_ns(self, ts: dict):
        try:
            return np.array([ts["s"] * 1e9 + ts["ns"]])
        except Exception as e:
            print(f"hex_ts_to_ns failed: {e}")
            return np.array([np.inf])
