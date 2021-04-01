---
layout: post
title:  "도커를 이용한 <br>GPU vs CPU 테스트"
subtitle: "GPU를 이용할 수 있는 도커 / 테스트 / 테스트 결과"
date:  2021-04-01 13:00:00 +0900
background: '/img/post-001.jpg'
categories:
   - NLP 
tags:
   - "deeq NLP"
   - 디큐 NLP
   - docker
   - gpu
   - cpu
   - python
   - grpc
   - baikalai
   - 바이칼AI
   - 자연어처리
   - 인공지능
   - natural language processing
   - 바이칼에이아이


author: 김진수 <jskim@baikal.ai>
---

# 개요

Baikal Ai 개발팀에서는 `NER(Named Entity Recognition)`, `TC(Text Classification)`의 모델링, 예측을 수행할 수 있도록 독립적인 서버를 구축했습니다.
그 서버는 메인서버의 뒷단에 위치할 계획이기 때문에 `BIB(Back In Backend)`라고 부릅니다.
BIB 에서는 모델링, 예측을 수행할 서버들이 작동될 것이기 때문에 BIB 서버를 도커라이징하여 개별 컨테이너로 관리했을 때 효율적인 관리가 가능할 것이라 예상하였습니다.
이번 포스팅에서는 BIB를 도커라이징하여 컨테이너를 생성한 다음 `GPU를 이용했을 때의 작업속도`와 `CPU를 이용했을 때의 작업속도`를 비교해보도록 하겠습니다.
이 과정에서 우리가 살펴보아야 할 점은 크게 두가지입니다.
첫 번째, GPU를 이용할 수 있는 도커 이미지를 만드는 간단한 방법 및 의의, 두 번째, GPU와 CPU를 이용했을 때 퍼포먼스의 차이가 될 것 같습니다.
물론 많은 사람들이 예상하다시피 GPU를 이용했을 때 당연히 더 빨리 작업이 진행될 것입니다.
GPU를 이용했을 때 빠르다면 얼마나 빠를 것인지 확인하는 차원에서 진행되었음을 알립니다.
테스트를 통해서 결과가 도출되면 효율적인 작업을 위해 어떤 선택을 해야할지 결정하는데 도움이 될 것이라 기대합니다.
테스트는 다음과 같이 진행될 것입니다.<br>

1. BIB 소개 및 도커라이징의 의의
2. 사용된 데이터셋
3. 테스트 실행
4. 테스트 결과
5. 결론

# SECTION 1 - BIB 소개 및 도커라이징
BIB 서버는 사용자의 요청에 따라 그 수가 유동적으로 변할 수 있도록 설계되었습니다.
그러한 장점을 잘 살리고 효율적으로 관리하기 위해서 BIB 서버를 하나하나 컨테이너로 생성하고 관리하게 될 것입니다.
먼저 GPU를 이용하는 컨테이너와 CPU를 이용하는 컨테이너를 준비합니다.<br>

1. [`tensorflow/tensorflow:2.3.0-gpu`](https://www.tensorflow.org/install/docker?hl=ko){: target="_blank"}
2. [`python:3.6`](https://hub.docker.com/_/python){: target="_blank"}


도커 베이스 이미지는 위 두개를 각각 사용하였습니다.
텐서플로에서 제공하는 GPU 도커 이미지를 이용하면 별도의 설정 없이도 도커에서 GPU를 이용할 수 있습니다.
연결된 링크를 통해서 공식 가이드와 버전을 확인하실 수 있습니다.
물론 이미지를 빌드하고나면 도커 이미지의 사이즈가 상당히 큰 것을 볼 수 있는데 이러한 문제는 저희에게는 필요없는 텐서플로 패키지까지 포함되어 있기 때문입니다.
간단한 기능 테스트와 퍼포먼스 체크를 위해서 선택한 방법이니 도커를 통해 GPU를 이용하는 다른 방법을 사용하여 경량화 하셔도 무방합니다.
실제로 서비스를 준비하는 과정에서는 필수 데이터만으로 구성된 이미지를 만들 계획입니다.<br>

우선 도커 이미지를 확인합니다. 도커 이미지는 다음 명령어로 확인할 수 있습니다.

```shell
docker images
```

도커 이미지를 확인하고 위 두개의 도커 이미지가 없는 경우 `docker pull` 을 이용해서 직접 다운로드 받으셔도 되고 `docker run` 명령어를 이용할 때
로컬 환경에 해당 이미지가 존재하지 않으면 자동으로 다운받기 때문에 곧바로 컨테이너를 띄우셔도 됩니다.
베이스 이미지로 사용될 두개의 도커 이미지를 준비하셨다면 이제 저희가 원하는대로 이미지를 빌드할 차례입니다.
`Dockerfile`을 생성하실 때 다음과 같이 시작하시면 베이스 이미지로 시작할 수 있습니다.

```dockerfile
# gpu를 사용할 도커 이미지를 빌드하는 경우
FROM tensorflow/tensorflow:2.3.0-gpu
# 나머지 원하는 작업...

# gpu를 사용하지 않을 도커 이미지를 빌드하는 경우
FROM python3.6
# 나머지 원하는 작업...
```

만들어 둔 `Dockerfile`을 이용해서 도커 이미지를 빌드하였습니다.
도커 이미지를 확인해보면 아래와 같이 두개의 도커 이미지가 준비되었음을 확인할 수 있습니다.


![docker_images]({{"/assets/images/posts/Docker-In-GPU-VS-CPU/docker_images.png"| relative_url}}){: width="100%"}{: .center}

다음과 같은 명령어를 이용해서 4개의 각기 다른 컨테이너를 띄워줍니다.
```shell
docker run -d -p 20001:50051 --name bib_train_gpu bib_gpu
docker run -d -p 20002:50051 --name bib_predict_gpu bib_gpu
docker run -d -p 20003:50051 --name bib_train_gpu bib_cpu
docker run -d -p 20004:50051 --name bib_predict_gpu bib_cpu
```
도커 컨테이너들이 아래와 같이 생성되었습니다.

![docker_containers]({{"assets/images/posts/Docker-In-GPU-VS-CPU/docker_containers.png"| relative_url}}){: width="100%"}{: .center}

저희는 지금까지의 과정을 통해서 두가지 효과를 기대하고 있습니다.

1. BIB 서버 환경구축의 편리성
2. BIB 서버의 확장성 증대

첫 번째 효과는 도커를 사용하시는 분들은 누구든지 알고 계실 것이라 생각합니다.
BIB 서버를 <u>도커가 설치된 어디서든 이미지만 있다면 금방 띄울 수 있다는 것</u>은 업무 효율을 높여줍니다.
두 번째 효과로 확장성 증대라 함은 서비스의 규모에 따라 BIB 서버의 개수를 조절하기 용이하다는 것입니다.
예를 들어 일부 서버에 문제가 발생하는 경우에도 BIB 서버를 자동적으로 관리하는 로직에 따라
<u>문제가 발생한 일부 서버를 대체할 수 있는 새로운 서버를 바로 구성</u>할 수 있습니다.


# SECTION 2 - 사용된 데이터셋
이번 테스트에서 사용한 데이터셋은 국립국어원에서 제공하는 모두의 말뭉치와 자체적으로 수집한 데이터입니다.
모두의 말뭉치의 데이터를 가공하여 BIB에서 인식할 수 있도록 가공하여 NER 데이터셋으로 이용하였고
자체적으로 수집한 데이터를 이용하여 TC 데이터셋으로 이용하였습니다.
이번 테스트는 속도와 관련된만큼 사용된 데이터는 데이터의 크기를 중점적으로 고려하여 진행하였습니다.
하지만 모델을 학습하는 경우에는 데이터의 질에 따라 정확도 등이 크게 영향을 받게 됩니다.
따라서 이 테스트에서 학습된 모델의 정확도는 결과 지표로써 의미가 없다고 볼 수 있습니다.
<br>

1. [ner_train.txt]({{"/assets/downloads/Docker-In-GPU-VS-CPU/ner_train.txt"| relative_url}}){: download="ner_train.txt"}
2. [ner_predict.txt]({{"/assets/downloads/Docker-In-GPU-VS-CPU/ner_predict.txt"| relative_url}}){: download="ner_predict.txt"}
3. [tc_train.txt]({{"/assets/downloads/Docker-In-GPU-VS-CPU/inputs/tc_train.txt"| relative_url}}){: download="tc_train.txt"}
4. [tc_predict.txt]({{"/assets/downloads/Docker-In-GPU-VS-CPU/tc_predict.txt"| relative_url}}){: download="tc_predict.txt"}

# SECTION 3 - 테스트 실행
BIB 서버는 GRPC 서비스를 제공하는 GRPC 서버입니다.
따라서 리퀘스트를 보내기 위해서는 서버가 서비스하는 proto와 동일한 proto로 GRPC 클라이언트를 준비해야 합니다.
다음 예시는 bib_train_gpu 컨테이너에 TC 학습요청을 보내는 클라이언트 예시입니다.
[python 을 이용한 grpc 서버/클라이언트](https://grpc.io/docs/languages/python/quickstart/){: target="_blank"} 
```python
# client.py
import train_pb2_grpc as pb
import train_pb2 as pb2

channel = grpc.insecure_channel('0.0.0.0:20001')
stub = pb.TrainProviderStub(channel)
data_set = [] # 생략
request = pb2.TrainTextClassificationModelRequest(
                model_name='tc_train',
                epochs=1,
                train_rate=0.8,
                data_set=data_set
            )
stub.TrainTextClassificationModel(request)
```

클라이언트를 통해 요청을 보내고 BIB 에서 작업을 처리한 이후 결과를 다음과 같이 파일로 저장합니다. 
결과는 elapsed_time 의 값에 주목하여 보시면 됩니다.
예측의 경우 한 가지 고려사항이 있습니다.
학습이 완료된 모델이 파일로 존재하고 이 모델을 토대로 예측 작업을 수행하는 경우 
모델파일을 메모리에 로드하는 시간이 필요합니다. 따라서

1. 모델을 로드하지 않고 예측을 요청하여 모델로딩부터 예측까지
2. 모델을 미리 로딩해둔 이후 예측만

위 두 가지 예측 결과를 구분하여 확인합니다.

1. [GPU : NER 모델링]({{"/assets/downloads/Docker-In-GPU-VS-CPU/ner_train_gpu.txt"| relative_url}}){: download="ner_train_gpu.txt"}
2. [GPU : NER 예측(로딩부터)]({{"/assets/downloads/Docker-In-GPU-VS-CPU/ner_predict_gpu_from_load.txt"| relative_url}}){: download="ner_predict_gpu_from_load.txt"}
3. [GPU : NER 예측(예측만)]({{"/assets/downloads/Docker-In-GPU-VS-CPU/ner_predict_gpu_after_load.txt"| relative_url}}){: download="ner_predict_gpu_after_load.txt"}
4. [GPU : TC 모델링]({{"/assets/downloads/Docker-In-GPU-VS-CPU/tc_train_gpu.txt"| relative_url}}){: download="tc_train_gpu.txt"}
5. [GPU : TC 예측(로딩부터)]({{"/assets/downloads/Docker-In-GPU-VS-CPU/tc_predict_gpu_from_load.txt"| relative_url}}){: download="tc_predict_gpu_from_load.txt"}
6. [GPU : TC 예측(예측만)]({{"/assets/downloads/Docker-In-GPU-VS-CPU/tc_predict_gpu_after_load.txt"| relative_url}}){: download="tc_predict_gpu_after_load.txt"}
7. [CPU : NER 모델링]({{"/assets/downloads/Docker-In-GPU-VS-CPU/ner_train_cpu.txt"| relative_url}}){: download="ner_train_cpu.txt"}
8. [CPU : NER 예측(로딩부터)]({{"/assets/downloads/Docker-In-GPU-VS-CPU/ner_predict_cpu_from_load.txt"| relative_url}}){: download="ner_predict_cpu_from_load.txt"}
9. [CPU : NER 예측(예측만)]({{"/assets/downloads/Docker-In-GPU-VS-CPU/ner_predict_cpu_after_load.txt"| relative_url}}){: download="ner_predict_cpu_after_load.txt"}
10. [CPU : TC 모델링]({{"/assets/downloads/Docker-In-GPU-VS-CPU/tc_train_cpu.txt"| relative_url}}){: download="tc_train_cpu.txt"}
11. [CPU : TC 예측(로딩부터)]({{"/assets/downloads/Docker-In-GPU-VS-CPU/tc_predict_cpu_from_load.txt"| relative_url}}){: download="tc_predict_cpu_from_load.txt"}
12. [CPU : TC 예측(예측만)]({{"/assets/downloads/Docker-In-GPU-VS-CPU/tc_predict_cpu_after_load.txt"| relative_url}}){: download="tc_predict_cpu_after_load.txt"}


# SECTION 4 - 테스트 결과


파일로 확인할 수 있는 테스트 결과는 작업의 스텝별로 진행율을 나타내고 있기 때문에 테스트 목적에 맞게 시간 경과만 도식화해서 살펴봅시다.
테스트 결과는 GPU, CPU의 사양에 따라 다를 수 있습니다.


#### 1. 모델링

| |GPU|CPU|CPU / GPU|
|---|---|---|---|
|NER 작업시간|786.5초|2548.3초|3.24|
|TC 작업시간|170.1초|2173.3초|12.78|

- NER : GPU에서 작업하는 경우 CPU에 비해서 속도가 `3.24`배 가량 빠름.
- TC : GPU에서 작업하는 경우 CPU에 비해서 속도가 `12.78`배 가량 빠름.

#### 2. 예측

| |GPU|CPU|CPU / GPU|
|---|---|---|---|
|NER 작업시간(로딩부터)|66.9초|105.6초|1.58|
|NER 작업시간(예측만)|58.2초|99.5초|1.71|
|TC 작업시간(로딩부터)|15.4초|79.0초|5.13|
|TC 작업시간(예측만)|7.5초|72.2초|9.62|

- NER : GPU에서 작업하는 경우 CPU에 비해서 속도가 `1.58 ~ 1.71`배 가량 빠름.
- TC : GPU에서 작업하는 경우 CPU에 비해서 속도가 `5.13 ~ 9.62`배 가량 빠름.

# SECTION 5 - 결론

마지막으로 위의 테스트 결과를 이용해서 짧게 결론을 내려보겠습니다.
속도 차이에 관한 내용은 의견이 아닌 사실만 언급하였습니다.

#### BIB 서버를 도커화하여 사용하는 것의 장점

> BIB 서버를 도커 이미지로 빌드하여 관리하는 방식으로는 언제 어디서든 도커를 이용해서 관리할 수 있기 때문에 업무효율을 높일 수 있습니다.
> 또한 서비스 과정에서 일부 서버에서 문제가 발생했을 때 손쉽게 대체될 수 있기 때문에 고가용성을 제공합니다.

#### GPU VS CPU 속도 차이

> 예상했던대로 GPU를 이용하는 편이 훨씬 빠르게 작업이 진행됩니다.
> TC, NER 중에서 TC 에서 훨씬 큰 속도 차이를 보여주었고 학습된 모델을 이용하여 예측하는 과정보다
> 모델을 학습할 때 더 큰 속도 차이를 보여주었습니다.
