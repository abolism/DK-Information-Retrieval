from normalizer import PersianNormalizer
from data_loader import load_products, load_train_pairs, load_test_queries

normalizer = PersianNormalizer()
products = load_products('data/products.csv', normalizer)
print(products.head(1).to_dict(orient='records')[0])
train = load_train_pairs('data/train_query_product_pairs.csv')
print(train.head())
test = load_test_queries('data/test_queries.csv', normalizer)
print(test.head())