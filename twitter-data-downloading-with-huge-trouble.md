# Twitter Data Downloading with Huge Trouble

As the world's leading social media platform, people's sentiment reflected by twitter's historical data of some topics may affect the stock market. To achieve the research goal, we tried to download twitter historical data.

The special network condition of China mainland, together with the volatility of twitter API, make the process full of difficulties.

## 1 Package Install

### 1.1 [GetOldTweets3](https://github.com/Mottl/GetOldTweets3)

Both commands works

- pip install GetOldTweets3
- pip install -e git+https://github.com/Mottl/GetOldTweets3#egg=GetOldTweets3

Example code

- Get the last 2 tweets of the Barack Obama Whitehouse account

```python
import GetOldTweets3 as got

tweetCriteria = got.manager.TweetCriteria().setUsername("barackobama whitehouse")\
                                           .setMaxTweets(2)
tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]
print(tweet.text)
```

### 1.2 [twint](https://github.com/twintproject/twint)

Both commands work. But the second one is recommended since a newer version of the application will be installed.

- pip3 install twint
- pip3 install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint

Attention:

- [Git](https://git-scm.com/downloads) is required to be installed before the second install way
- [pipenv](https://github.com/twintproject/twint/issues/1039) is required to be installed 
- [Microsoft Visual C++ 14.0](https://github.com/twintproject/twint/issues/661) is required to be installed if run twint on Windows
- [nes_asyncio](https://github.com/twintproject/twint/issues/1121) is required if runing code in Spyder

Example code:

```python
import twint
import os

c = twint.Config()
c.Username = "noneprivacy"
c.Limit = 100
c.Store_csv = True
c.Output = "none.csv"
c.Lang = "en"
c.Translate = True
c.TranslateDest = "it"

twint.run.Search(c)
```

## 2 Resolve Network Limit

To make the Python package got access to Twitter successfully from China mainland, we tried many methods. All the solutions below are with pre-condition that a VPN is already started locally with SOCKS5 proxy running in port 7891 and HTTP proxy running in port 7890. If you are outside mainland China, this should not be a problem.

### 2.1 GetOldTweets3

For GetOldTweets3 to access Twitter, just set environment variables as below:

- For Windows:

  ```shell
  set http_proxy=http://127.0.0.1:7890
  set HTTP_PROXY=http://127.0.0.1:7890
  
  set https_proxy=http://127.0.0.1:7890
  set HTTPS_PROXY=http://127.0.0.1:7890
  ```

- For Linux/OS X

  ```shell
  export http_proxy=http://127.0.0.1:7890
  export HTTP_PROXY=http://127.0.0.1:7890
  
  export https_proxy=http://127.0.0.1:7890
  export HTTPS_PROXY=http://127.0.0.1:7890
  ```

We can also achieve this in python code like this:

```python
import os

proxy = 'http://127.0.0.1:7890'

os.environ['http_proxy'] = proxy 
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy
```

### 2.2 [twint](https://github.com/twintproject/twint)

The settings in 2.1 do not work for Twint. Twint is a more mature application that has its API for [proxy settings](https://github.com/twintproject/twint/issues/1097), like code below:

```python
c = twint.Config()
c.Username= ""
c.Store_object = True
c.Limit = 100
c.Proxy_host = "127.0.0.1"
c.Proxy_port = 7891
c.Proxy_type = "Socks5"
```

## 3 Volatility of Twitter API

Twitter [deletes their existed APIs](https://github.com/twintproject/twint/issues/915) occassionally, both [GetOldTweets3](https://github.com/Mottl/GetOldTweets3/issues/101) and [twint](https://github.com/twintproject/twint/issues/1119) were not working since twitter's changes on https://twitter.com/i/search/timeline page. It no longer exists since 0:00 on September 18, 2020.

Twint [fixed this](https://github.com/twintproject/twint/issues/915#issuecomment-716196991) quickly while GetOldTweets3 did not do so until now.

Twint did a good job on the above problem. But we quickly found that the proxy settings of it mentioned above just not worked and there were users reported this [problem](https://github.com/twintproject/twint/issues/1124) 5 days ago and waiting to be resolved.

## 4 Summary

Because of the volatility of Twitter API, GetOldTweets3 is disabled now. Twint fixed this in time but got proxy functionality not worked for no reason. We will ask our team members resident in HK for help to test the Twint without the network problem and also we will have a look at other Twitter scrapers like [Scweet](https://github.com/Altimis/Scweet), which is not that mature but may work better to twitter's current API.
