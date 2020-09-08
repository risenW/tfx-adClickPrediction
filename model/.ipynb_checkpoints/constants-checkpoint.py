# FEATURE DEFINITIONS FOR ADVERT-PRED

DENSE_FLOAT_FEATURE_KEYS = ['DailyTimeSpentOnSite', 'Age','AreaIncome', 'DailyInternetUsage' ]

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
VOCAB_SIZE = 1000

# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
OOV_SIZE = 10

VOCAB_FEATURE_KEYS = ['City', 'Male', 'Country' ]


# Keys
LABEL_KEY = 'ClickedOnAd'


def transformed_name(key):
    return key + '_xf'