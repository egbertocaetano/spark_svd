from pyspark.ml.classification import LinearSVC
from pyspark.mllib.linalg.distributed import IndexedRowMatrix, IndexedRow
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from scipy import sparse
import numpy as np


def indexed_row_matrix_to_numpy_matrix(mtx, shape: tuple):
    """

    :param mtx:
    :param shape
    :return:
    """

    rs = mtx.rows.map(lambda r: (r.index, r.vector.array.tolist()))

    _mtx = np.empty(list(shape), dtype=np.float64)

    _rows = rs.collect()

    # TODO: Check shape before insert into matrix

    for idx, row in _rows:

        _mtx[idx] = row

    return _mtx


spark_conf = SparkConf()
spark_builder = (SparkSession.builder.appName("test_log"))
spark_builder.config(conf=spark_conf)
spark = spark_builder.getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.sparkContext.setLogLevel("ERROR")

path = "/home/forrest/workspace/LINE/Baselines/AMR/results/19-05-23__23-07-42__MSRParaphraseCorpus/matrix/document-concept-matrix.npz"

# Load training data
# training = spark.read.format("libsvm").load(path)

sc = spark.sparkContext

doc_conc_mtx = sparse.load_npz(path)

doc_conc_mtx = doc_conc_mtx.todense()

shape = doc_conc_mtx.shape

indexed_doc_concept = [IndexedRow(idx, doc_conc_mtx[idx].tolist()[0]) for idx in range(0, shape[0])]

# indexed_sample = [IndexedRow(idx, doc_conc_list[idx]) for idx in range(0, len(sample_list))]

rows = sc.parallelize(indexed_doc_concept)

matrix = IndexedRowMatrix(rows)

del doc_conc_mtx, indexed_doc_concept

np_matrix = indexed_row_matrix_to_numpy_matrix(matrix, (11604, 14428))

print(np_matrix.shape)
# svd = mtx.computeSVD(k=100)



"""
lsvc = LinearSVC(maxIter=10, regParam=0.1)

# Fit the model
lsvcModel = lsvc.fit(training)

# Print the coefficients and intercept for linearsSVC
print("Coefficients: " + str(lsvcModel.coefficients))
print("Intercept: " + str(lsvcModel.intercept))
"""
