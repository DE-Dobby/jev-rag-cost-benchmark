# GCP 설정 가이드

인프라 변경(API 활성화, IAM, 서비스 계정/버킷/Job 생성)은 Claude가 직접 실행하지 않는다.
여기 정리된 `gcloud` 명령어를 사용자가 직접 복사해서 실행한다.

`<PROJECT_ID>`, `<BILLING_ACCOUNT_ID>` 등 꺾쇠괄호는 본인 값으로 바꿔서 실행할 것.

## 1. 지금 실행 (파일럿/로컬 개발용 — 최소 세팅)

로컬에서 파일럿(30건)과 본 실행 전 개발을 하려면 이것만 있으면 된다.
Secret Manager / Cloud Storage / Cloud Run Job은 아직 필요 없음 (3번 참고).

```bash
# 프로젝트 설정
gcloud config set project <PROJECT_ID>

# Vertex AI API 활성화 (Gemini 호출에 필요)
gcloud services enable aiplatform.googleapis.com

# 로컬 개발용 ADC 인증 (서비스 계정 키 파일 없이 내 계정으로 인증)
gcloud auth application-default login

# ADC가 사용할 quota project 지정 (위 로그인 후 1회)
gcloud auth application-default set-quota-project <PROJECT_ID>
```

확인 완료: `gemini-2.5-pro`, `gemini-2.5-flash` 모두 us-west1(Oregon)에서 서빙됨
(출처: cloud.google.com/vertex-ai/generative-ai/docs/learn/locations, 확인일 2026-10-04).
참고로 `gcloud ai models list`는 Model Registry에 올린 커스텀 모델만 보여주고
Gemini 같은 1st-party 모델은 안 나오므로 리전 확인에는 쓸 수 없음.

## 2. Billing 예산 알림 설정 (지금 실행 권장)

API 호출이 시작되기 전에 설정해두는 걸 권장.

```bash
gcloud billing budgets create \
  --billing-account=<BILLING_ACCOUNT_ID> \
  --display-name="jev-rag-cost-benchmark budget" \
  --budget-amount=<원하는 USD 금액> \
  --threshold-rule=percent=0.5 \
  --threshold-rule=percent=0.9 \
  --threshold-rule=percent=1.0
```
(세부 알림 채널(이메일 등)은 Console의 Billing > Budgets & alerts에서 추가 설정 가능)

## 3. 나중에 실행 — Phase 3 본 실행 준비 (아직 실행하지 말 것)

- [ ] Secret Manager API 활성화 + `JEV_API_KEY` 시크릿 등록
- [ ] Cloud Run Job용 서비스 계정 생성 + 최소 권한 IAM (Vertex AI 사용자, Secret Manager 접근자, Storage 객체 관리자)
- [ ] Cloud Storage 버킷 생성 (캐시 / 원본 응답 / 결과)
- [ ] Cloud Run Job 정의 및 배포

실제 명령어는 Phase 3 진입 직전, provider 어댑터와 캐시 스키마가 확정된 뒤 채운다.
