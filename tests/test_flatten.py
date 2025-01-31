import pandas as pd
import pytest

from pd_flatten import pd_flatten

pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 50)
pd.set_option("display.max_info_columns", 30)
pd.set_option("display.max_info_rows", 20)
pd.set_option("display.max_rows", 20)
pd.set_option("display.max_seq_items", None)
pd.set_option("display.width", 200)
pd.set_option("expand_frame_repr", True)
pd.set_option("mode.chained_assignment", "warn")


class TestNesting:
    def test_single_nested(self):
        df = pd.DataFrame([{"a": 0, "b": {"i": 1, "j": 2}}])

        observed = pd_flatten(df, name_columns_with_parent=False)
        expected = pd.DataFrame([{"a": 0, "i": 1, "j": 2}])

        pd.testing.assert_frame_equal(observed, expected)

    def test_double_nested(self):
        df = pd.DataFrame([{"a": 0, "b": {"i": 1, "j": {"k": 2, "l": 3}}}])

        observed = pd_flatten(df, name_columns_with_parent=False)
        expected = pd.DataFrame([{"a": 0, "i": 1, "k": 2, "l": 3}])

        pd.testing.assert_frame_equal(observed, expected)


class TestSep:
    def test_single_nested(self):
        df = pd.DataFrame([{"a": 0, "b": {"i": 1, "j": 2}}])

        observed = pd_flatten(df, name_columns_with_parent=True, sep="__")
        expected = pd.DataFrame([{"a": 0, "b__i": 1, "b__j": 2}])

        pd.testing.assert_frame_equal(observed, expected)

    def test_double_nested(self):
        df = pd.DataFrame([{"a": 0, "b": {"i": 1, "j": {"k": 2, "l": 3}}}])

        observed = pd_flatten(df, name_columns_with_parent=True, sep=".")
        expected = pd.DataFrame([{"a": 0, "b.i": 1, "b.j.k": 2, "b.j.l": 3}])

        pd.testing.assert_frame_equal(observed, expected)


class TestListExplosion:
    def test_single_nested_list(self):
        df = pd.DataFrame([{"a": 0, "b": [{"i": 1}, {"i": 2}]}])

        observed = pd_flatten(df, name_columns_with_parent=False)
        expected = pd.DataFrame([{"a": 0, "i": 1}, {"a": 0, "i": 2}])

        pd.testing.assert_frame_equal(observed, expected)

    def test_mixed_nested_list(self):
        df = pd.DataFrame([{"a": 0, "b": [{"i": 1}, {"j": 3}]}])

        observed = pd_flatten(df, name_columns_with_parent=False)
        expected = pd.DataFrame(
            [{"a": 0, "i": 1, "j": None}, {"a": 0, "i": None, "j": 3}]
        )

        pd.testing.assert_frame_equal(observed, expected)

    def test_single_nested_list_with_empty(self):
        df = pd.DataFrame([{"a": 0, "b": [{"i": 1}, None]}])

        observed = pd_flatten(df, name_columns_with_parent=False)
        expected = pd.DataFrame([{"a": 0, "i": 1}, {"a": 0, "i": None}])

        pd.testing.assert_frame_equal(observed, expected)


class TestExcludedColumns:
    def test_single_nested(self):
        df = pd.DataFrame([{"a": 0, "b": {"i": 1, "j": 2}}])

        observed = pd_flatten(df, name_columns_with_parent=False, except_cols=["b"])
        expected = df

        pd.testing.assert_frame_equal(observed, expected)

    def test_double_nested(self):
        df = pd.DataFrame([{"a": 0, "b": {"i": 1, "j": {"k": 2, "l": 3}}}])

        observed = pd_flatten(df, name_columns_with_parent=False, except_cols=["j"])
        expected = pd.DataFrame([{"a": 0, "i": 1, "j": {"k": 2, "l": 3}}])

        pd.testing.assert_frame_equal(observed, expected)


class TestDupColumnNames:
    def test_ok_when_naming_with_parent(self):
        df = pd.DataFrame([{"a": 0, "b": {"i": 1, "a": 2}}])

        observed = pd_flatten(df, name_columns_with_parent=True, sep="__")
        expected = pd.DataFrame([{"a": 0, "b__i": 1, "b__a": 2}])

        pd.testing.assert_frame_equal(observed, expected)

    def test_eror_when_naming_without_parent(self):
        df = pd.DataFrame([{"a": 0, "b": {"i": 1, "a": 2}}])

        with pytest.raises(
            NameError, match="Column names {'a'} on the column path `b` are duplicated"
        ):
            _ = pd_flatten(df, name_columns_with_parent=False)


class TestGraphqlOutput:
    def test_real_example(self):
        df = pd.DataFrame(
            [
                {
                    "model_id": "ACH-000625",
                    "cell_line_name": "Hep 3B2.1-7",
                    "stripped_cell_line_name": "HEP3B217",
                    "model_conditions": [
                        {
                            "model_condition_id": "MC-000625-ULD9",
                            "omics_profiles": [
                                {
                                    "profile_id": "PR-pyUsq4",
                                    "datatype": "wes",
                                    "blacklist_omics": False,
                                    "omics_order_date": None,
                                    "smid_ordered": None,
                                    "smid_returned": None,
                                    "omics_sequencings": [
                                        {
                                            "blacklist": False,
                                            "expected_type": "wes",
                                            "sequencing_id": "CDS-ly9ev0",
                                            "source": "SANGER",
                                            "version": 1,
                                            "sequencing_alignments": [
                                                {
                                                    "id": 5845,
                                                    "url": "gs://bucket-name/wes/SANGER_HEP3B217_LIVER.bam",
                                                    "index_url": "gs://bucket-name/wes/SANGER_HEP3B217_LIVER.bai",
                                                    "size": 10689474743,
                                                    "reference_genome": "hg19",
                                                    "sequencing_alignment_source": "GP",
                                                },
                                                {
                                                    "id": 12871,
                                                    "url": "gs://bucket-name/hg38_wes/CDS-ly9ev0.hg38.bam",
                                                    "index_url": "gs://bucket-name/hg38_wes/CDS-ly9ev0.hg38.bai",
                                                    "size": 5269538310,
                                                    "reference_genome": "hg38",
                                                    "sequencing_alignment_source": "CDS",
                                                },
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                        {
                            "model_condition_id": "MC-000625-ENPZ",
                            "omics_profiles": [
                                {
                                    "profile_id": "PR-illyAu",
                                    "datatype": "wes",
                                    "blacklist_omics": False,
                                    "omics_order_date": None,
                                    "smid_ordered": None,
                                    "smid_returned": None,
                                    "omics_sequencings": [
                                        {
                                            "blacklist": False,
                                            "expected_type": "wes",
                                            "sequencing_id": "CDS-Mzr7zM",
                                            "source": "CCLE2",
                                            "version": 2,
                                            "sequencing_alignments": [
                                                {
                                                    "id": 5940,
                                                    "url": "gs://bucket-name/wes/C836.Hep_3B2.1-7.1.bam",
                                                    "index_url": "gs://bucket-name/wes/C836.Hep_3B2.1-7.1.bai",
                                                    "size": 9692030267,
                                                    "reference_genome": "hg19",
                                                    "sequencing_alignment_source": "GP",
                                                },
                                                {
                                                    "id": 12963,
                                                    "url": "gs://bucket-name/hg38_wes/CDS-Mzr7zM.hg38.bam",
                                                    "index_url": "gs://bucket-name/hg38_wes/CDS-Mzr7zM.hg38.bai",
                                                    "size": 5172718041,
                                                    "reference_genome": "hg38",
                                                    "sequencing_alignment_source": "CDS",
                                                },
                                            ],
                                        }
                                    ],
                                },
                                {
                                    "profile_id": "PR-Y5zRFS",
                                    "datatype": "RRBS",
                                    "blacklist_omics": False,
                                    "omics_order_date": None,
                                    "smid_ordered": None,
                                    "smid_returned": None,
                                    "omics_sequencings": [
                                        {
                                            "blacklist": False,
                                            "expected_type": "RRBS",
                                            "sequencing_id": "CDS-raqeR0",
                                            "source": "CCLE2",
                                            "version": 1,
                                            "sequencing_alignments": [
                                                {
                                                    "id": 3614,
                                                    "url": "gs://bucket-name/RRBS/G29750/Hep_3B2.1-7/v2/Hep_3B2.1-7.bam",
                                                    "index_url": "gs://bucket-name/RRBS/G29750/Hep_3B2.1-7/v2/Hep_3B2.1-7.bai",
                                                    "size": 1550416155,
                                                    "reference_genome": "hg19",
                                                    "sequencing_alignment_source": "GP",
                                                }
                                            ],
                                        }
                                    ],
                                },
                            ],
                        },
                    ],
                }
            ]
        )

        observed = pd_flatten(df, name_columns_with_parent=False)

        expected = pd.DataFrame(
            [
                {
                    "model_id": "ACH-000625",
                    "cell_line_name": "Hep 3B2.1-7",
                    "stripped_cell_line_name": "HEP3B217",
                    "model_condition_id": "MC-000625-ULD9",
                    "profile_id": "PR-pyUsq4",
                    "datatype": "wes",
                    "blacklist_omics": False,
                    "omics_order_date": None,
                    "smid_ordered": None,
                    "smid_returned": None,
                    "blacklist": False,
                    "expected_type": "wes",
                    "sequencing_id": "CDS-ly9ev0",
                    "source": "SANGER",
                    "version": 1,
                    "id": 5845,
                    "url": "gs://bucket-name/wes/SANGER_HEP3B217_LIVER.bam",
                    "index_url": "gs://bucket-name/wes/SANGER_HEP3B217_LIVER.bai",
                    "size": 10689474743,
                    "reference_genome": "hg19",
                    "sequencing_alignment_source": "GP",
                },
                {
                    "model_id": "ACH-000625",
                    "cell_line_name": "Hep 3B2.1-7",
                    "stripped_cell_line_name": "HEP3B217",
                    "model_condition_id": "MC-000625-ULD9",
                    "profile_id": "PR-pyUsq4",
                    "datatype": "wes",
                    "blacklist_omics": False,
                    "omics_order_date": None,
                    "smid_ordered": None,
                    "smid_returned": None,
                    "blacklist": False,
                    "expected_type": "wes",
                    "sequencing_id": "CDS-ly9ev0",
                    "source": "SANGER",
                    "version": 1,
                    "id": 12871,
                    "url": "gs://bucket-name/hg38_wes/CDS-ly9ev0.hg38.bam",
                    "index_url": "gs://bucket-name/hg38_wes/CDS-ly9ev0.hg38.bai",
                    "size": 5269538310,
                    "reference_genome": "hg38",
                    "sequencing_alignment_source": "CDS",
                },
                {
                    "model_id": "ACH-000625",
                    "cell_line_name": "Hep 3B2.1-7",
                    "stripped_cell_line_name": "HEP3B217",
                    "model_condition_id": "MC-000625-ENPZ",
                    "profile_id": "PR-illyAu",
                    "datatype": "wes",
                    "blacklist_omics": False,
                    "omics_order_date": None,
                    "smid_ordered": None,
                    "smid_returned": None,
                    "blacklist": False,
                    "expected_type": "wes",
                    "sequencing_id": "CDS-Mzr7zM",
                    "source": "CCLE2",
                    "version": 2,
                    "id": 5940,
                    "url": "gs://bucket-name/wes/C836.Hep_3B2.1-7.1.bam",
                    "index_url": "gs://bucket-name/wes/C836.Hep_3B2.1-7.1.bai",
                    "size": 9692030267,
                    "reference_genome": "hg19",
                    "sequencing_alignment_source": "GP",
                },
                {
                    "model_id": "ACH-000625",
                    "cell_line_name": "Hep 3B2.1-7",
                    "stripped_cell_line_name": "HEP3B217",
                    "model_condition_id": "MC-000625-ENPZ",
                    "profile_id": "PR-illyAu",
                    "datatype": "wes",
                    "blacklist_omics": False,
                    "omics_order_date": None,
                    "smid_ordered": None,
                    "smid_returned": None,
                    "blacklist": False,
                    "expected_type": "wes",
                    "sequencing_id": "CDS-Mzr7zM",
                    "source": "CCLE2",
                    "version": 2,
                    "id": 12963,
                    "url": "gs://bucket-name/hg38_wes/CDS-Mzr7zM.hg38.bam",
                    "index_url": "gs://bucket-name/hg38_wes/CDS-Mzr7zM.hg38.bai",
                    "size": 5172718041,
                    "reference_genome": "hg38",
                    "sequencing_alignment_source": "CDS",
                },
                {
                    "model_id": "ACH-000625",
                    "cell_line_name": "Hep 3B2.1-7",
                    "stripped_cell_line_name": "HEP3B217",
                    "model_condition_id": "MC-000625-ENPZ",
                    "profile_id": "PR-Y5zRFS",
                    "datatype": "RRBS",
                    "blacklist_omics": False,
                    "omics_order_date": None,
                    "smid_ordered": None,
                    "smid_returned": None,
                    "blacklist": False,
                    "expected_type": "RRBS",
                    "sequencing_id": "CDS-raqeR0",
                    "source": "CCLE2",
                    "version": 1,
                    "id": 3614,
                    "url": "gs://bucket-name/RRBS/G29750/Hep_3B2.1-7/v2/Hep_3B2.1-7.bam",
                    "index_url": "gs://bucket-name/RRBS/G29750/Hep_3B2.1-7/v2/Hep_3B2.1-7.bai",
                    "size": 1550416155,
                    "reference_genome": "hg19",
                    "sequencing_alignment_source": "GP",
                },
            ]
        )

        pd.testing.assert_frame_equal(observed, expected)

    def test_one_with_empty_list(self):
        df = pd.DataFrame(
            [
                {
                    "model_id": "ACH-003093",
                    "cell_line_name": "SU-DIPG-6",
                    "stripped_cell_line_name": "SUDIPG6",
                    "model_conditions": [
                        {
                            "model_condition_id": "MC-003093-nD4R",
                            "omics_profiles": [],
                        },
                        {
                            "model_condition_id": "MC-003093-jlhL",
                            "omics_profiles": [
                                {
                                    "profile_id": "PR-Cx75w1",
                                    "datatype": "wgs",
                                    "blacklist_omics": False,
                                    "omics_order_date": "2023-07-31",
                                    "smid_ordered": "SM-NA1VM",
                                    "smid_returned": "SM-MME96,SM-NG7S5",
                                    "omics_sequencings": [
                                        {
                                            "blacklist": False,
                                            "expected_type": "wgs",
                                            "sequencing_id": "CDS-3wbI7z",
                                            "source": "DEPMAP",
                                            "version": 1,
                                            "sequencing_alignments": [
                                                {
                                                    "id": 8891,
                                                    "url": "gs://bucket-name/wgs_hg38_cram/CDS-3wbI7z.cram",
                                                    "index_url": "gs://bucket-name/wgs_hg38_cram/CDS-3wbI7z.crai",
                                                    "size": 19972004183,
                                                    "reference_genome": "hg38",
                                                    "sequencing_alignment_source": "GP",
                                                },
                                                {
                                                    "id": 15823,
                                                    "url": "gs://bucket-name/wgs_hg38/CDS-3wbI7z.hg38.bam",
                                                    "index_url": "gs://bucket-name/wgs_hg38/CDS-3wbI7z.hg38.bai",
                                                    "size": 81978444448,
                                                    "reference_genome": "hg38",
                                                    "sequencing_alignment_source": "CDS",
                                                },
                                            ],
                                        }
                                    ],
                                },
                                {
                                    "profile_id": "PR-GIe5GE",
                                    "datatype": "rna",
                                    "blacklist_omics": False,
                                    "omics_order_date": "2023-07-31",
                                    "smid_ordered": "SM-NA1VM",
                                    "smid_returned": None,
                                    "omics_sequencings": [
                                        {
                                            "blacklist": False,
                                            "expected_type": "rna",
                                            "sequencing_id": "CDS-MaS6dz",
                                            "source": "DEPMAP",
                                            "version": 1,
                                            "sequencing_alignments": [
                                                {
                                                    "id": 4771,
                                                    "url": "gs://bucket-name/rna/CDS-MaS6dz.bam",
                                                    "index_url": "gs://bucket-name/rna/CDS-MaS6dz.bai",
                                                    "size": 5955045320,
                                                    "reference_genome": "hg19",
                                                    "sequencing_alignment_source": "GP",
                                                },
                                                {
                                                    "id": 11876,
                                                    "url": "gs://bucket-name/rnasq_hg38/CDS-MaS6dz.Aligned.sortedByCoord.out.bam",
                                                    "index_url": "gs://bucket-name/rnasq_hg38/CDS-MaS6dz.Aligned.sortedByCoord.out.bam.bai",
                                                    "size": 4967381407,
                                                    "reference_genome": "hg38",
                                                    "sequencing_alignment_source": "CDS",
                                                },
                                            ],
                                        }
                                    ],
                                },
                            ],
                        },
                    ],
                }
            ]
        )

        observed = pd_flatten(df, name_columns_with_parent=False)

        expected = pd.DataFrame(
            [
                {
                    "model_id": "ACH-003093",
                    "cell_line_name": "SU-DIPG-6",
                    "stripped_cell_line_name": "SUDIPG6",
                    "model_condition_id": "MC-003093-nD4R",
                    "profile_id": None,
                    "datatype": None,
                    "blacklist_omics": None,
                    "omics_order_date": None,
                    "smid_ordered": None,
                    "smid_returned": None,
                    "blacklist": None,
                    "expected_type": None,
                    "sequencing_id": None,
                    "source": None,
                    "version": None,
                    "id": None,
                    "url": None,
                    "index_url": None,
                    "size": None,
                    "reference_genome": None,
                    "sequencing_alignment_source": None,
                },
                {
                    "model_id": "ACH-003093",
                    "cell_line_name": "SU-DIPG-6",
                    "stripped_cell_line_name": "SUDIPG6",
                    "model_condition_id": "MC-003093-jlhL",
                    "profile_id": "PR-Cx75w1",
                    "datatype": "wgs",
                    "blacklist_omics": False,
                    "omics_order_date": "2023-07-31",
                    "smid_ordered": "SM-NA1VM",
                    "smid_returned": "SM-MME96,SM-NG7S5",
                    "blacklist": False,
                    "expected_type": "wgs",
                    "sequencing_id": "CDS-3wbI7z",
                    "source": "DEPMAP",
                    "version": 1,
                    "id": 8891,
                    "url": "gs://bucket-name/wgs_hg38_cram/CDS-3wbI7z.cram",
                    "index_url": "gs://bucket-name/wgs_hg38_cram/CDS-3wbI7z.crai",
                    "size": 19972004183,
                    "reference_genome": "hg38",
                    "sequencing_alignment_source": "GP",
                },
                {
                    "model_id": "ACH-003093",
                    "cell_line_name": "SU-DIPG-6",
                    "stripped_cell_line_name": "SUDIPG6",
                    "model_condition_id": "MC-003093-jlhL",
                    "profile_id": "PR-Cx75w1",
                    "datatype": "wgs",
                    "blacklist_omics": False,
                    "omics_order_date": "2023-07-31",
                    "smid_ordered": "SM-NA1VM",
                    "smid_returned": "SM-MME96,SM-NG7S5",
                    "blacklist": False,
                    "expected_type": "wgs",
                    "sequencing_id": "CDS-3wbI7z",
                    "source": "DEPMAP",
                    "version": 1,
                    "id": 15823,
                    "url": "gs://bucket-name/wgs_hg38/CDS-3wbI7z.hg38.bam",
                    "index_url": "gs://bucket-name/wgs_hg38/CDS-3wbI7z.hg38.bai",
                    "size": 81978444448,
                    "reference_genome": "hg38",
                    "sequencing_alignment_source": "CDS",
                },
                {
                    "model_id": "ACH-003093",
                    "cell_line_name": "SU-DIPG-6",
                    "stripped_cell_line_name": "SUDIPG6",
                    "model_condition_id": "MC-003093-jlhL",
                    "profile_id": "PR-GIe5GE",
                    "datatype": "rna",
                    "blacklist_omics": False,
                    "omics_order_date": "2023-07-31",
                    "smid_ordered": "SM-NA1VM",
                    "smid_returned": None,
                    "blacklist": False,
                    "expected_type": "rna",
                    "sequencing_id": "CDS-MaS6dz",
                    "source": "DEPMAP",
                    "version": 1,
                    "id": 4771,
                    "url": "gs://bucket-name/rna/CDS-MaS6dz.bam",
                    "index_url": "gs://bucket-name/rna/CDS-MaS6dz.bai",
                    "size": 5955045320,
                    "reference_genome": "hg19",
                    "sequencing_alignment_source": "GP",
                },
                {
                    "model_id": "ACH-003093",
                    "cell_line_name": "SU-DIPG-6",
                    "stripped_cell_line_name": "SUDIPG6",
                    "model_condition_id": "MC-003093-jlhL",
                    "profile_id": "PR-GIe5GE",
                    "datatype": "rna",
                    "blacklist_omics": False,
                    "omics_order_date": "2023-07-31",
                    "smid_ordered": "SM-NA1VM",
                    "smid_returned": None,
                    "blacklist": False,
                    "expected_type": "rna",
                    "sequencing_id": "CDS-MaS6dz",
                    "source": "DEPMAP",
                    "version": 1,
                    "id": 11876,
                    "url": "gs://bucket-name/rnasq_hg38/CDS-MaS6dz.Aligned.sortedByCoord.out.bam",
                    "index_url": "gs://bucket-name/rnasq_hg38/CDS-MaS6dz.Aligned.sortedByCoord.out.bam.bai",
                    "size": 4967381407,
                    "reference_genome": "hg38",
                    "sequencing_alignment_source": "CDS",
                },
            ]
        )

        pd.testing.assert_frame_equal(observed, expected)
