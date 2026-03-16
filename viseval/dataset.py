# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from pathlib import Path


class Dataset:
    def __init__(
        self,
        folder: Path,
        table_type: str = "all",
        with_irrelevant_tables: bool = False,
        limit: int = None,
        db_id: str = None,
    ):
        self.folder = folder
        dict_name = "visEval"
        if table_type in ["single", "multiple"]:
            dict_name += "_" + table_type
        dict_name += ".json"
        with open(folder / dict_name) as f:
            self.dict = json.load(f)

        with open(folder / "databases/db_tables.json") as f:
            self.db_tables = json.load(f)

        def benchmark():
            keys = list(self.dict.keys())
            # 如果指定了 db_id，先过滤出匹配的实例
            if db_id is not None:
                keys = [key for key in keys if self.dict[key].get("db_id") == db_id]
            if limit is not None:
                keys = keys[:limit]
            for key in keys:
                self.dict[key]["id"] = key
                self.dict[key]["tables"] = self.__get_tables(
                    key, with_irrelevant_tables
                )
                yield self.dict[key]

        self.benchmark = benchmark()

    def __get_tables(self, id: str, with_irrelevant_tables: bool = False):
        spec = self.dict[id]
        db_id = spec["db_id"]
        # table name
        all_table_names = self.db_tables[db_id]
        table_names = [
            x
            for x in all_table_names
            if x.lower() in spec["vis_query"]["VQL"].lower().split()
        ]

        if with_irrelevant_tables:
            irrelevant_tables = spec["irrelevant_tables"]
            table_names.extend(irrelevant_tables)

        tables = list(
            map(
                lambda table_name: f"{self.folder}/databases/{db_id}/{table_name}.csv",
                table_names,
            )
        )

        return tables
