from pyspark import SparkContext, SparkConf
from pyspark.storagelevel import StorageLevel
import json
import re
import heapq

# ---------- Configuration ----------
conf = (
    SparkConf()
    .setAppName("ChiSquare_TopTerms_Final")
    .set("spark.driver.memory", "6g")
    .set("spark.executor.memory", "7g")
    .set("spark.executor.cores", "2")
    .set("spark.default.parallelism", "96")
)

sc = SparkContext(conf=conf)

# ---------- Input Paths ----------
input_path = "hdfs:///user/dic25_shared/amazon-reviews/full/reviewscombined.json"
stopwords_file = "stopwords.txt"

# ---------- Load the stopwords ----------
with open(stopwords_file, 'r') as f:
    stopwords = set([line.strip().lower() for line in f])

# ---------- Tokenization, stopword, and short word filtering ----------
def preprocess(text):
    tokens = re.split(r"[ \t\d\(\)\[\]\{\}\.!?,;:+=\-_\"'`~#@&*%€$§\\/]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in stopwords]

# ---------- Read and preprocess the reviews (keep only those that have review text and category) ----------
reviews = sc.textFile(input_path) \
    .map(lambda line: json.loads(line)) \
    .filter(lambda x: 'reviewText' in x and 'category' in x) \
    .map(lambda x: (x['category'], preprocess(x['reviewText'])))

# Persist to disk (tried with cache first, but it did not work)
reviews.persist(StorageLevel.DISK_ONLY)

# ---------- Count the documents ----------
total_docs = reviews.count()

docs_per_category = reviews.map(lambda x: (x[0], 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .collectAsMap()

docs_per_category_bc = sc.broadcast(docs_per_category)

# ---------- Count the terms ----------
term_category_pairs = reviews.flatMap(
    lambda x: [((x[0], term), 1) for term in set(x[1])]
)

A_counts = term_category_pairs.reduceByKey(lambda a, b: a + b)

# Filter terms that only occur once (this improved runtime and also produced an identical output as the code without it so it is not wrong)
A_counts = A_counts.filter(lambda x: x[1] >= 2)

# ---------- Frequencies of global terms ----------
term_global = reviews.flatMap(
    lambda x: [(term, 1) for term in set(x[1])]
)

term_doc_freq = term_global.reduceByKey(lambda a, b: a + b).collectAsMap()
term_doc_freq_bc = sc.broadcast(term_doc_freq)

# ---------- Calculate the Chi-Squares ----------
def chi_square_calc(category_term_count):
    (category, term), A = category_term_count
    B = docs_per_category_bc.value[category] - A
    C = term_doc_freq_bc.value[term] - A
    D = total_docs - (A + B + C)

    N = A + B + C + D
    numerator = (A * D - B * C) ** 2 * N
    denominator = (A + B) * (C + D) * (A + C) * (B + D)
    chi2 = numerator / denominator if denominator != 0 else 0.0
    return (category, (term, chi2))

chi_square_rdd = A_counts.map(chi_square_calc)

# ---------- Top 75 per category ----------
def top75_combiner(term_chi):
    return [term_chi]

def top75_merge_value(heap, term_chi):
    heapq.heappush(heap, term_chi)
    if len(heap) > 75:
        heapq.heappop(heap)
    return heap

def top75_merge_combiners(heap1, heap2):
    combined = heap1 + heap2
    return heapq.nlargest(75, combined, key=lambda x: x[1])

top_terms = chi_square_rdd.combineByKey(
    top75_combiner,
    top75_merge_value,
    top75_merge_combiners
)

# ---------- Format the output file ----------
category_lines = top_terms.map(
    lambda x: f"{x[0]} " + " ".join(
        [f"{term}:{chi2:.4f}" for term, chi2 in sorted(x[1], key=lambda y: -y[1])]
    )
)

# ---------- Add the Dictionary ----------
merged_dict = top_terms.flatMap(
    lambda x: [term for term, _ in x[1]]
).distinct()

# ---------- Save the output ----------
category_lines_list = category_lines.collect()
merged_list = merged_dict.sortBy(lambda x: x).collect()
merged_line = " ".join(merged_list)

# Prepare a list with all lines: category lines first, then the merged dictionary line (as is in Assignment 1)
all_output_lines = sorted(category_lines_list) + [merged_line]

# Turns terms and dictionary and converts to RDD so spark can write it in parallel and into 1 file
sc.parallelize(all_output_lines).coalesce(1).saveAsTextFile("output_rdd_final")




