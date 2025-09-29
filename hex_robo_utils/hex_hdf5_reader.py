#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-09-19
################################################################

import time
import h5py
import numpy as np


class HexHdf5Reader:

    def __init__(self, file_path: str):
        self.__file_path = file_path
        self.__hdf5_file = h5py.File(file_path, "r")
        self.__dataset_dict = {}

    def __del__(self):
        if hasattr(self, '__hdf5_file') and self.__hdf5_file is not None:
            self.__hdf5_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.__hdf5_file is not None:
            self.__hdf5_file.close()
            self.__hdf5_file = None

    def get_shape(self, group_name: str, dataset_name: str = "data"):
        dataset = self.__get_dataset_handle(group_name, dataset_name)
        return dataset.shape

    def get_dtype(self, group_name: str, dataset_name: str = "data"):
        dataset = self.__get_dataset_handle(group_name, dataset_name)
        return dataset.dtype

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

    def get_data(
        self,
        group_name: str,
        index: int,
        dataset_name: str = "data",
    ):
        return self.get_batch_data(
            group_name=group_name,
            start_index=index,
            end_index=index + 1,
            dataset_name=dataset_name,
        )[0]

    def get_all_data(
        self,
        group_name: str,
        dataset_name: str = "data",
    ):
        return self.get_batch_data(
            group_name=group_name,
            start_index=0,
            end_index=None,
            dataset_name=dataset_name,
        )

    def get_batch_data(
        self,
        group_name: str,
        start_index: int,
        end_index: int = None,
        dataset_name: str = "data",
    ):
        dataset = self.__get_dataset_handle(group_name, dataset_name)

        if start_index < 0:
            start_index = 0
        elif start_index >= dataset.shape[0]:
            raise IndexError(
                f"Start index {start_index} out of range [0, {dataset.shape[0]})"
            )

        if end_index is None or end_index > dataset.shape[0]:
            end_index = dataset.shape[0]
        elif end_index < start_index:
            end_index = start_index

        data = dataset[start_index:end_index]
        if dataset_name != "data":
            if data.shape[1:] != (1, ):
                raise ValueError(
                    f"Dataset {group_name}/{dataset_name} is not a timestamp dataset (expected shape[1:] = (1,), got {data.shape[1:]})"
                )
            ts_list = []
            for ts_ns in data.reshape(-1):
                ts_list.append(self.__ns_to_hex_ts(int(ts_ns)))
            return ts_list
        return data

    def __ns_to_hex_ts(self, ts: int):
        return {
            "s": ts // 1_000_000_000,
            "ns": ts % 1_000_000_000,
        }

    def now_ns(self):
        return np.array([time.time_ns()])

    def hex_ts_to_ns(self, ts: dict):
        try:
            return np.array([ts["s"] * 1e9 + ts["ns"]])
        except Exception as e:
            print(f"hex_ts_to_ns failed: {e}")
            return np.array([np.inf])
