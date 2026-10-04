# GCP 설정 가이드

인프라 변경(API 활성화, IAM, 서비스 계정/버킷/Job 생성)은 Claude가 직접 실행하지 않는다.
이 문서에 필요한 `gcloud` 명령어를 정리해두고, 사용자가 직접 실행한다.

실제 명령어는 Phase 2(provider 어댑터 확정 — 어떤 API/권한이 필요한지 정해진 뒤)와
Phase 3(Cloud Run Job 본 실행 준비) 단계에서 채운다. 현재는 자리표시자.

## 예정 항목
- [ ] 필요한 API 활성화 목록 (Vertex AI, Secret Manager, Cloud Storage, Cloud Run)
- [ ] 서비스 계정 생성 및 최소 권한 IAM 바인딩
- [ ] Secret Manager에 `JEV_API_KEY` 등록
- [ ] Cloud Storage 버킷 생성 (캐시/원본 응답/결과)
- [ ] Cloud Run Job 정의 및 배포
- [ ] Billing 예산 알림 설정 (README 안내와 연계)
