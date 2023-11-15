python perspective_api_span.py \
--file ../dataset/span_cnn_train.json \
--output ../dataset/span_cnn_train_score.json \
--api_key <your_api_key> \
--api_rate <your_api_rate> \
--process 100

python perspective_api_span.py \
--file ../dataset/span_cnn_test.json \
--output ../dataset/span_cnn_test_score.json \
--api_key <your_api_key> \
--api_rate <your_api_rate> \
--process 100