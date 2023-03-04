# 络打油

（神经网）络打油 - 基于 chatGPT API 的罗大佑风格歌词生成器。

用法：

```
usage: lyrics_master.py [-h] [-c] [-d] [-t TEMPERATURE] [-p TOP_P] [subject]

自动生成罗大佑风格的歌词。 使用前请将 OPENAI_API_KEY 环境变量设成您的 openAI API key 值： export
OPENAI_API_KEY=<您的 openAI API key>

positional arguments:
  subject               歌曲的主题

optional arguments:
  -h, --help            show this help message and exit
  -c, --chatgpt         用 chatGPT 模型来产生歌词
  -d, --davinci         用 Davinci 模型来产生歌词
  -t TEMPERATURE, --temperature TEMPERATURE
                        想象力有多狂野 ([0, 1]区间的实数)
  -p TOP_P, --top_p TOP_P
                        只考虑头部概率的选择 ([0, 1]区间的实数)
```
