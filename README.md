# 20_1_GoogleColab
## Google colab이란?
머신러닝을 처음 입문할 때, 첫 난관은 개발 환경을 구축하는 것입니다.

어마어마하게 비싼 하드웨어 가격은 물론이며, Cuda와 같이 생소한 놈들과 다양한 파이썬 라이브러리들을 만나야 하죠.

Colab은 이러한 문제를 해결해줍니다. "쥬피터 노트북"을 처음 사용해보신 분들이라면 Colab에 접속하는 순간 친근한 느낌을 받습니다.
왜냐햐면, 쥬피터 개발진들이 참여해섬 만들었습니다!!

### 1. Colab 시작하기
Colab은 쥬피터 서버와 유사하지만 몇 가지 알고가야할 부분이 있습니다.
* 최대 12시간만 작업이 가능합니다.
  * Learning 상태로 방치해두면 중간에 커널이 죽은것을 볼 수 있습니다. 따라서 중간중간 weight와 loss를 저장하는 코드를 넣어야합니다.
* 하드웨어 가속시(GPU / TPU) 를 제공합니다.
  * [수정 -> 노트 설정] 또는 [런타임 -> 런타임 유형 변경 ] 에서 설정 가능
  * 암호화 화폐 채굴 등을 수행할 경우는 서비스 차단이 이루어질수도 있음
  * TPU는 머신러닝을 위해 Google에서 자체적으로 제작한 하드웨어로 GPU보다 월등한 성능을 보인다.
  * 실제 GPU, TPU 차이를 테스트를 해본적은 없지만, Keras 라이브러리는 TPU를 자동으로 잡지 못하기 때문에 별도 코드를 추가해줘야한다.
  * GPU의 경우 약 14.6 GB 크기를 제공해준다.
* 구글 드라이비와 연동이 가능하다.

### 1.1 Colaboratory 환경
* CPU : Intel (R) Xeon(R) CPU @ 2.30 GHz (Dual-Core)
* GPU : Nvidia Tesla K80
* GPU Memory : 13GB
* VM 지속시간 : 12시간

무려 Xeon CPU에 Tesla GPU 환경을 이용할 수 있다는 것이 가장 큰 매력!!

실제 CPU에서만 training 해볼때와 GPU boosting 했을 때 그 차이가 10~45배 정도 난다.

직접 테스트 해볼 수 있는 소스코드는 아래!

```python
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
   raise SystemError('GPU device not found')
print ('Found GPU at: {}'.format(device_name))
```
* GPU vs TPU 성능 차이 
```python
import tensorflow as tf
import timeit

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.device('/cpu:0'):
    random_image_cpu  = tf.random_normal((100, 100, 100, 3))
    net_cpu = tf.layers.conv2d(random_image_cpu, 32, 7)
    net_cpu = tf.reduce_sum(net_cpu)

with tf.device('/gpu:0'):
    random_image_gpu = tf.random_normal((100, 100, 100, 3))
    net_gpu = tf.layers.conv2d(random_image_gpu, 32, 7)
    net_gpu = tf.reduce_sum(net_gpu)

sess = tf.Session(config=config)

# Test execution once to detect errors early.
try:
  sess.run(tf.global_variables_initializer())
except tf.errors.InvalidArgumentError:
  print(
      '\n\nThis error most likely means that this notebook is not '
      'configured to use a GPU.  Change this in Notebook Settings via the '
      'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
  raise

def cpu():
  sess.run(net_cpu)
  
def gpu():
  sess.run(net_gpu)
  
# Runs the op several times.
print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
      '(batch x height x width x channel). Sum of ten runs.')
print('CPU (s):')
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))

sess.close()

```

### 2. Colab 알아보기
* 단축키 설정하기 : [도구 -> 단축키] 에서 원하는 단축기 설정이 가능
* 테마 : [도구 -> 환경설정] 에서 'Dark' 등 3가지 선택 가능
* Github 백업하기 : [파일 -> Giuhub에 사본 저장]을 통해 Github와 연동하여 사본을 저장 가능
* 입력효과 넣기 : [도구 -> 환경설정-> 기타] 에서 파워레벨 효과 설정에 따라 입력시 불꽃이 튀기는 효과를 줄 수 있음
* 리눅스 OS 버전 : Ubuntu 18.04 를 사용중

### 2.1 버전, 하드웨어 사양 (CPU, RAM , DISK)
```bash
# Version
!cat /etc/issue.net

# CPU
!head /proc/cpuinfo

# RAM
!head -n 3 /proc/meminfo

# DISK
!df -h
```

### 2.2 GPU 사양
```bash
# GPU
!nvidia-smi
```

### 3. Github에 커밋된 notebook을 colab에서 바로 열고 실행하기
Github에서 커밋된 소스를 colab에서 매우 쉽게 바로 열 수 있습니다.

<strong>주소</strong>만 바꾸어 주면 됩니다.

예를 들어,
https://github.com/jaedeokhan/20_1_AI_Introduce/pratice/0326THU_IntroDL_mnist.ipynb

https://<strong>colab.research.google.com/github/</strong>jaedeokhan/20_1_AI_Introduce/<strong>blob</strong>/master/pratice/0326THU_IntroDL_mnist.ipynb

다음과 같이 colab.research.google.com/github , blob만 수정을 해주면 된다.

### 3.1 구글 코랩(Google colab)의 런타임 연결 끊김을 방지하는 방법
구글 코랩(Google colab)은 90분 동안 아무런 interaction이 없거나, 총 12시간의 세션 timeout이 존재한다.

더 쉽게 설명하자면, colab에서 학습을 돌려놓고 90분 동안 브라우저에 아무런 interaction을 주지 않는다면, 긴 학습을 다 수행하지 못하고 끊겨버릴 수 있다.

이를 방지하기 위해서는 javascript 코드를 소개하고, 이 코드를 console 창에 실행시킴으로써 90분의 idle timeout을 방지하는 방법을 소개

> 1. chorme F12 개발자 콘솔 열기
개발자 콘솔에서 가장 밑에 코드를 입력할 수 있는 창에다가 다음과 같이 입력하기

```javascript
function ClickConnect() {
    console.log("코랩 연결 끊김 방지");
    document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect, 60 * 1000)
```
그러면, 매 1분마다 콘솔창에 "코랩 연결 끊김 방지" 문구와 함께 <storng>colab idle timeout이 되는 것을 방지</strong>

### 3.2 Google Colab에서 python 패키지를 영구적(permanetly)으로 설치하는 방법
Colab에서 영구적(permanetly)으로 파이썬 패키지를 설치(install)하는 방법에 대해 공유

Google Colab은 정말 좋은 무료 환경입니다. 꽤 좋은 성능을 내는 하드웨어 자원을 무료로 사용할 수 있습니다. 때문에, 머신러닝/딥러닝 을 학습하시는 분들께는 정말 좋은 환경이라 최근 사용하시는 분들이 급격히 늘어나고 있습니다.

pre-built되는 library가 있지만, 항상 최신 버전으로 지원하고 있지 않기 때문에,

때론 !pip install 패키지 명령어를 매번 실행해주면서 패키지를 업그레이드 해줘야할 필요가 있습니다.

* Google Colab에서 permanent하게 패키지를 설치하기
이제부터 제가 알려드리는 방식대로 진행하시면, colab을 새로 열어도 자동으로 제가 설치한 패키지를 가져올 수 있습니다.

아래의 방식은 다른 파이썬 패키지도 동일하게 적용할 수 있습니다.

설치된 패키지를 구글 드라이브에 저장을 하게 되는데요. 저장할 폴더를 먼저 만들어 줍니다.

저는 Colab Notebooks에 my_env라는 폴더를 만들어 주고 패키지를 그곳에 저장하도록 하겠습니다.












