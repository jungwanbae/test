# QUAD 모의투자 단계별 가이드

국내 레그(MSAP-M + 고금리 대비형 코스피 부분)와 미국 레그(확인형 역발상 VT + 레버리지 RP + 고금리 대비형 미국 부분)를
**돈이 오가지 않는 상태**에서 끝까지 돌려 보는 순서입니다. 모든 단계는 기본값이 모의투자(`broker.env: mock`)입니다.

## 0. 준비물
- 상시 켜 둘 수 있는 PC(또는 소형 클라우드 서버). 미국 레그는 **한국시간 밤 11시~새벽**에 주문합니다.
- Python 3.10 이상, Git, 디스크 여유 2GB(국내 전종목 데이터 약 520MB 포함)
- NH투자증권 계좌(나무)

## 1. NH 나무 PLUG 신청
1. https://www.nhplug.com 에서 Open API 사용 신청 → **APP KEY / APP SECRET** 발급
2. 같은 곳에서 **모의투자 신청**(모의 계좌 = acct_type 03, 가상 자금 지급)
3. 해외주식도 API 로 쓰려면 해외주식 거래 서비스가 열려 있어야 합니다(앱/홈페이지에서 확인)
4. 시세 조회는 NH 정책상 운영 서버에서만 되므로, 모의투자여도 같은 키로 실시세를 받습니다(프로그램이 자동 처리)

## 2. 설치
```bash
git clone https://github.com/jungwanbae/test.git autotrader && cd autotrader
git checkout claude/auto-trading-software-0iq1f6
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[report,dev]"
pytest -q                              # 88 passed 가 나오면 설치 정상
```

## 3. 비밀값(.env)
```bash
cp .env.example .env                   # Windows: copy .env.example .env
```
`.env` 를 열어 `NHPLUG_APP_KEY`, `NHPLUG_APP_SECRET` 만 채웁니다. `NHPLUG_ACCOUNT` 는 **비워 두면 모의 계좌(03)를 자동 선택**합니다.
`.env` 는 절대 공유·커밋하지 마세요.

## 4. 연결 확인(주문 없음)
```bash
autotrader account                                   # 국내 모의 계좌: 주문가능현금·보유
autotrader --config config/triad_us.yaml account     # 해외 모의 계좌: 달러 주문가능금액·보유
```
- `오류: ... 계좌(acct_type ('03',))가 없습니다` → 모의투자 신청이 안 된 상태입니다.
- 해외 계좌의 **달러 주문가능금액이 0** 이면 → 6-B 의 페이퍼 모드로 미국 레그를 시험하세요.

## 5. 데이터 준비(최초 1회, 약 5분)
```bash
autotrader stock-data          # 국내 전종목 데이터 내려받기(약 520MB) + 학습 데이터셋
```
종목 모델(시드 3개 앙상블, 2026-11 학습본)은 저장소에 들어 있어 바로 쓸 수 있습니다. 매달 한 번 `autotrader stock-train` 으로 재학습하세요.

## 6. 신호·드라이런(주문 없음, 아무 때나)
```bash
autotrader signal                                    # 국내: 종목 15개 + ETF + 고금리 대비형 코스피 비중
autotrader --config config/triad_us.yaml signal      # 미국: QQQ·TQQQ·TLT·TMF·GLD·UGL·DBC·SGOV 비중, 변동성·부스트 상태
autotrader rebalance --dry-run --force               # 국내: 지금 잔고 기준 주문 계획만 출력
autotrader --config config/triad_us.yaml rebalance --dry-run --force
```

### 6-B. (해외 모의 계좌에 달러가 없을 때) 미국 레그 페이퍼 모드
`config/triad_us.yaml` 의 `broker.name` 을 `paper` 로 바꿉니다. NH 실시세로 가격을 받고 체결은 PC 안에서 가상으로 처리하며,
가상 자금 $50,000(`paper_initial_cash`)과 상태 파일 `state/paper_us.json` 을 씁니다.

## 7. 첫 모의 주문(장중에만)
| 레그 | 주문 가능 시간 | 명령 |
|---|---|---|
| 국내 | 평일 09:00~15:00 (한국시간) | `autotrader rebalance --force` |
| 미국 | 미국 장중 09:35~15:30 ET = 한국시간 **23:35~05:30**(11~3월) / **22:35~04:30**(3~11월 서머타임) | `autotrader --config config/triad_us.yaml rebalance --force` |

- `--force` 는 '이번 달 리밸런싱 날이 아니어도 지금 실행'입니다. 첫 실행에만 쓰세요.
- 주문은 매도 → 매수 순서로, 즉시 체결 지향 지정가 → 미체결 취소 → 마지막 라운드 시장가로 나갑니다.

## 8. 결과 확인
```bash
autotrader account                                   # 체결 후 잔고
autotrader status                                    # 최근 실행 이력(성공/부분체결/오류)
autotrader --config config/triad_us.yaml status
autotrader triad                                     # 국내 30% : 미국 70% 배분 점검 → 벗어나면 환전 금액 안내
```
로그는 `logs/`, 실행·주문 기록은 `state/autotrader.db`(국내)·`state/triad_us.db`(미국)에 남습니다.

## 9. 자동 실행(매 거래일)
터미널 두 개(또는 tmux/screen, Windows 는 창 두 개)를 띄워 둡니다.
```bash
autotrader daemon                                    # 국내: 매 거래일 09:30 KST
autotrader --config config/triad_us.yaml daemon      # 미국: 매 미국 거래일 10:30 ET
```
- 매일 점검하지만 실제 매매는 **월초 정기 리밸런싱 · 부스트 같은 신호 변경 · 비중 이탈 · 손절**일 때만 합니다.
- PC 절전/잠자기를 끄세요. 꺼져 있던 날은 다음 실행 때 자동으로 따라잡습니다(부분체결·오류는 같은 날 최대 3회 재시도).
- 텔레그램 알림: `.env` 에 `TELEGRAM_BOT_TOKEN`·`TELEGRAM_CHAT_ID`, 설정의 `notify.telegram: true`.

## 10. 안전장치·운영 수칙
- **긴급 정지**: 프로젝트 폴더에 `STOP` 파일을 만들면 즉시 매매 중단(`touch STOP`, Windows: `type nul > STOP`). 지우면 재개.
- 고금리 규칙 기준금리(현재 4.0%): 연준 금리가 바뀌면 `config/triad_us.yaml` 의 `cash_rate` 와
  `config/default.yaml` 의 `msap.hr_kospi.us_rate` 를 함께 고치세요.
- 기존 '고금리 대비형' 알림을 따로 따라 매매 중이면 같은 돈으로 이중 운용하지 마세요(끄려면 `hr_weight: 0`, `hr_kospi.weight: 0`).

## 11. 실전 전환 체크리스트(최소 1개월 모의 후)
1. 모의 기간 동안 `status` 에 오류·잔여 주문이 반복되지 않았는가
2. 신호(`signal`)가 백테스트 규칙대로 월초·사건일에만 바뀌었는가
3. 해외 레버리지 ETP·국내 레버리지 ETF 매매 요건(사전교육·기본예탁금 등)을 계좌에서 마쳤는가
4. 두 설정의 `broker.env: live`, `.env` 에 `AUTOTRADER_LIVE_CONFIRM=I_UNDERSTAND_THE_RISK`
5. 처음엔 `risk.capital_limit` 으로 소액만(예: 국내 3,000,000원 / 미국 $2,000) 배정한 뒤 단계적으로 늘리기
