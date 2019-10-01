import steamreviews
import json
import csv

request_params = dict()
request_params['language'] = 'english'

steamreviews.download_reviews_for_app_id_batch(chosen_request_params=request_params)

