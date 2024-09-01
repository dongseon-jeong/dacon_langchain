
## 1. 대회 목표


"재정기관의 서류를 참고하여 질문에 알맞은 답변을 생성하라"

## 2. 규칙


- 인풋 : 질문과 정보가 저장된 PDF파일 정보(파일명과 파일 저장 위치)를 함께 제공
- 모델 사용 제한 : 오픈 소스 모델만 가능
- 트레인 데이터 : 주어진 데이터는 질문과 PDF파일 정보, 정답으로 LLM 모델 학습에는 제한이 있음
- 평가 F1 스코어 : 
	- 정답과 같은 의미이나 정답보다 추가적인 단어가 포함될 경우 precision이 떨어짐
	- 재정 정보 특성상 표 데이터가 많아 정확한 표인식이 어려울 경우 recall이 떨어짐

## 3. 기획


### a. 참고
- 베이스 라인 코드는 PDF파일 정보 추출 및 faiss DB 생성 후 Langchain RAG 활용
- 다른 참가자들은 PDF파일에서 마크 다운 형식으로 데이터 추출 후 사용
- 여러 방안이 있음 : LLM 변경, LLM 온도 조정, LLM 시퀀스 길이 조정, 청크 크기 조정, 청크 오버랩 조정, 임베딩 모델 변경, 리트리버 변경,  리트리버 top n 조정, 리트리버 앙상블 활용, 리랭크 활용, 프롬프트 엔지니어링, 모델 파인튜닝 등
### b. 초기 기획 (Langchain 활용)
**멀티 모달 모델 활용**
![[Pasted image 20240828103757.png]](./image/Pasted%20image%2020240828103757.png)  
회색음영 : PDF to Vector DB
1. PDF에서 텍스트와 테이블 이미지 추출
2. Multi modal Model로 각 데이터 요약 후 임베딩 생성
3. 유저 인풋 입력
4. 리트리버가 멀티 벡터 DB에서 유사한 인덱스 확인하여 매칭 데이터 추출
5. 프롬프트에 추가 후 Multi modal Model에 입력
6. 아웃풋 생성

### c. 최종 (Langchain 활용)
**OCR 활용**
![[Pasted image 20240828102335.png]](./image/Pasted%20image%2020240828102335.png)  
회색음영 : PDF to Vector DB
1. PDF에서 텍스트 추출, PDF 페이지별 이미지 변환
2. 각 이미지에서 테이블 이미지 추출 후 OCR로 텍스트 추출 > JSON 형태의 텍스트로 변환
3. 각 텍스트 임베딩 생성
4. 유저 인풋 입력 > LLM을 통한 질문 이진 분류
5. 이진 분류 결과에 해당하는 리트리버 선택 후 벡터 DB에서 유사한 인덱스 확인하여 매칭 데이터 추출
6. 프롬프트에 추가 후 LLM에 입력
7. 아웃풋 생성

## 4. 멀티 모달 파인튜닝


### 참고
- 고려할 수 있는 방향
	- 한국어로 파인 튜닝된 멀티모달 모델 사용
		- kollava : 정황 설명 가능하나 이미지 상 한국어를 잘 인식하지 못함 > 파인튜닝 필요
			~~~
			kollava로 영문 table VQA 데이터 학습 테스트 :
			transformers.trainer api에서 알 수 없는 오류 발생
			구조상 동일한 llava는 오류 없이 학습 가능하여 시간 관계상 다른 방법 시도
			~~~
	- 영어 멀티모달 모델 아웃풋을 LLM모델 인풋에 사용
		- Florence, layoutLM, PaliGemma... : vision 한국어 인식 못함 > 파인튜닝 필요
			~~~
			Florence 한국어 OCR 데이터 학습 테스트 :
			글자 인식뿐만 아니라 Bbox 인식도 기대치에 미치지 못함
			~~~
	- 한국어 LLM모델과 한국어 Vision모델을 프로젝션 레이어를 통해 사용
		- kor llama3 + sigLIP : vision 한국어 인식 못함 > 각 파인튜닝 필요
			~~~
			llama3 + sigLIP 영문 table VQA 데이터 학습 테스트 :
			정확도 점검 필요
			한글 VQA 학습 시 processor.tokenizer에 토큰 추가 필요
			~~~
- 참고 사항
	- 한국어 VQA 데이터가 없음(테이블 관련)
	- AI hub 공공기간 한국어 OCR 데이터 존재
	- huggingface 영문 table VQA 데이터 존재
- 멀티모달
	- joint embedding : add embedding > cosine similarity loss
	- projection layer : add embedding > cross entropy loss
	- cross attention : 쿼리 + 다른 모델 키 밸류

###  projection layer  학습
- **llava 소스 코드 및 llama3 multi modal 추론 코드 참고**
- 기본구조 ![[Pasted image 20240831173924.png]](./image/Pasted%20image%2020240831173924.png)  
	1. 이미지 > vision 모델 > projection 레이어> 이미지 임베딩 생성
	2. 텍스트 > langauge 모델 임베딩 > 텍스트 임베딩 생성
	3. 이미지 임베딩, 텍스트 임베딩 > concat 임베딩
	4. concat 임베딩 > langauge 모델 어텐션 레이어 입력 > 문장 생성
	5. 학습 과정을 통해 projection 레이어 weight 학습
- 모델 : vision-sigLIP, langauge-llama3
- 학습 데이터셋 : llava pretrain dataset
- loss function : cross entropy

프리트레인 데이터로 projection layer를 먼저 파인튜닝 후 테이블 QA 데이터로 재학습이 필요 
일정 이상 성능을 위해 많은 학습 시간 및 짜임새있는 학습 전략 필요함 > 보류

## 5. OCR 활용


테이블 이미지에서 JSON 형태의 표 데이터 추출
- 인풋(표이미지) :

| a   | b   | c   |
| --- | --- | --- |
| 1   | 2   | 3   |
| 4   | 5   | 6   |

- 아웃풋(텍스트) : "[['a','b','c'],[1,2,3],[4,5,6],.....]"

프롬프트 엔지니어링을 통해 행 머리글 및 열 머리글을 파악

실제 데이터는 공무원 특유의 끼워맞추기씩 테이블로 인해 아웃풋 데이터가 깔끔하지 않음
예시)
![[Pasted image 20240828133955.png]](./image/Pasted%20image%2020240828133955.png)  

Table Transformer
Paddle OCR
