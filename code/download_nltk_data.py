import nltk
import argparse

def download(proxy):

  if (proxy != None):
    nltk.set_proxy(proxy)
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')
  else:
    try:
      nltk.download('stopwords')
      nltk.download('punkt')
      nltk.download('averaged_perceptron_tagger')
      nltk.download('wordnet')
      nltk.download('vader_lexicon')
    except:
        print("Download error, enable proxy and try again")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download nltk data.")

    parser.add_argument('--proxy', type=str, help='Proxy used for data downloading, like http://127.0.0.1:7890', default=None)

    args = parser.parse_args()
    proxy = args.proxy

    download(proxy)