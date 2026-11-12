# Northwood 성능 기준선

기준 장비는 Lenovo IdeaPad 1 15IGL7(82V7, Celeron N4120, 4 GiB)과 RAM 8 GiB 이상의 x86-64 UEFI PC 한 대다. 두 장비의 첫 기준선은 아직 채워지지 않았다. 15IGL7에서 게스트를 시작하고 로그를 회수할 수 있는 빌드가 필요하다.

## 측정 시나리오

| ID | 시작 | 종료 및 관찰 |
| --- | --- | --- |
| S1 | 전원을 켜고 호스트 진단 선택 | 진단 완료 또는 거부. RAM 계측, CPU 기능, 저장장치와 화면 모드 기록 |
| S2 | 설치 미디어로 부팅 | 첫 설치 화면이 표시되고 입력을 받음. 첫 프레임과 입력 가능 시점을 따로 기록 |
| S3 | 같은 설치 화면에서 설치 시작 | 설치 완료 후 첫 설정 화면. 네트워크 대기와 재부팅을 포함하고 중간 재부팅 횟수 기록 |
| S4 | 설치된 볼륨을 선택해 부팅 | 데스크톱에서 입력을 받음. 로그인 절차를 두 빌드에서 동일하게 유지 |
| S5 | 데스크톱에서 유휴 측정 시작 | 5분 경과. 자동 색인 등 관찰한 백그라운드 작업 기록 |
| S6 | 같은 유휴 구간을 마친 뒤 Safari 실행 | 같은 페이지의 본문 표시 및 스크롤 가능. 비교에서는 페이지, 브라우저 캐시 상태, 네트워크 조건 고정 |

같은 UEFI 설정, 전원 모드, USB와 포트, macOS 빌드, 설치 볼륨, 게스트 PE 수를 사용한다. 기준선은 기본 설정으로 실행한다. `--profile`은 핫 블록과 상세 타이밍을 수집하는 별도 실행에 사용한다. 번역 팩을 비교하는 경우 각 실행의 캐시 초기 상태와 전원 종료 방식을 기록한다. 실기 부팅 시간은 조건마다 3회 측정하고 개별 값과 중앙값을 보존한다. Apple Silicon의 QEMU TCG 실행에서는 카운터, 종료 사유, 화면 흐름을 비교한다. 처리율과 절대 시간은 실기에서 비교한다.

## 로그와 시간 기록

- 실행마다 로그 원본과 `boot-log --perf` 요약을 보존한다. 매체에서 회수한 로그가 여러 번의 부팅을 담으면 시나리오의 시작과 끝에 해당하는 부팅을 식별한다. 로그가 중간부터 시작하거나 끝이 잘렸으면 비교표에 표시한다.
- `STEP <이름> t=<초.밀리초>`와 `STEP o/w/f <이름> ms=<밀리초>`는 단계의 시작과 소요 시간을 기록한다. 기존 로그에는 번들 읽기, 번역 캐시 열기, 프로세서 인계, 디스크·USB 시작, macOS 시작 단계가 있다. 프로세서 인계 단계에는 `ExitBootServices` 전후의 작업이 함께 들어간다.
- `BOOT STAGE name=first_gop_frame`의 시각은 GOP에 첫 완성 프레임이 기록된 시점이다. 설치 화면이나 데스크톱의 입력 가능 시점은 별도로 기록한다.
- 전원 버튼부터 시작하는 시간, 설치 화면의 입력 가능 시점, 설치 완료, 데스크톱 준비, Safari 응답 시점은 별도 스톱워치나 영상으로 기록한다. 한 시나리오의 구간 차이는 같은 시계에서 계산한다. TSC 출처와 보정 정보는 `HOST TSC`에서 기록한다.
- `BEAT`의 `compiled`는 보고 구간마다 합산한다. `discards`, `evictions`, `exhausted`는 부팅 시작부터의 누적값이므로 구간의 끝과 시작의 차이를 쓴다. `HOST SERVICE`의 `longest_us display`, `total_us display`, `maximum_gap_us`와 `BEAT COST us`는 같은 시나리오의 보고 구간끼리 비교한다. `GUEST PERFORMANCE`의 처리율은 같은 수의 워밍업 구간을 뺀 뒤 비교한다.
- `HOST MEMORY code=`는 러너별 사설 영역을 포함한 전체 예약량이다. 코드 윈도우 비교에는 `PES CODE`의 공유 윈도우 크기, 링 세대, 잠금 대기와 재설치 분류를 사용한다.
- CSV의 `PES CODE`에서 `board/window`는 공유 윈도우 크기, `board/generation`은 해당 시점의 링 세대다. `churn/ring`과 `lock/waited`는 직전 샘플 이후의 재설치 횟수와 대기 시간이다. 동일한 시나리오에서 링 세대는 구간의 끝과 시작의 차이를, 재설치와 대기는 구간 합계를 비교한다.

```sh
stela boot-log --perf disk.img
stela boot-log --perf --text serial.log
stela boot-log --perf --text --csv serial.log > performance.csv
```

장비별 측정은 `hardware/perf/<장비>/<날짜>-<커밋>.md`에 둔다. 기록에는 장비 모델과 RAM, UEFI 버전, 부트 바이너리 SHA-256, macOS 빌드, 스토리지와 번역 팩 상태, 옵션, 시나리오, 개별 실행 결과, 로그 경로를 포함한다. 장치 일련번호나 비공개 로그는 공유 기록에서 제거한다.

## QEMU 단계 계측 확인

2026-11-12. 동일한 EFI `7a857694fc4840eb626a2cbb1affbb6a9e6bd4f8ff5aeb1252687c5d009c3065`, 4 GiB VM, macOS 26B123. 각 열은 TCG 단일 실행이다. 부팅 구간의 계측 위치를 확인하는 데 썼다.

| 부팅 단계 | 전체 CPU 기능 (ms) | AVX·BMI 비활성 (ms) |
| --- | ---: | ---: |
| 부트 파일 읽기 | 677 | 647 |
| 번역 캐시 열기 | 2 | 2 |
| 호스트 메모리 준비 | 857 | 746 |
| 디스크·USB 시작 | 796 | 818 |
| macOS 메모리 적재 | 7149 | 6816 |
| 부트 PE 진입, 로더 시계 기준 누적 | 14026 | 13686 |
| 마지막으로 확인한 게스트 단계 | XNU | SPTM |

배포형 EFI `75d58793…`의 별도 4 GiB·4코어 QEMU TCG에서 부트 PE 진입은 로더 시계 기준 20,626 ms였다. 첫 NVRAM 동기화의 시작과 정상 반환은 게스트 24 MHz 카운터로 363.538초 간격이었다. 두 로그 줄 사이에 포함된 호스트 단계 세부 표본 341개의 합은 345,801 ms다. `launchd`의 early boot 완료부터 `restored_external` PID 5 생성까지는 게스트 로그 시계로 40.836초였다. 데몬 생성 뒤에도 USB 포트 1의 DWC3 전원 도메인은 꺼져 있었다. 데몬이 요청한 두 번째 NVRAM 동기화도 같은 게스트 카운터로 442.943초 후 정상 반환됐고, 이후 `mount_tmpfs /mnt5`가 시작됐다.

별도 4 GiB 통합 EFI `14f73ff1…` 실행에서 첫 NVRAM 동기화는 게스트 카운터로 404.258초 후 성공했다. `restored_external` PID 5가 실행된 뒤에도 DWC3와 PHY의 차단 도메인이 남아 있었다. 두 번째 동기화는 402.667초 후 성공했고, 임시 파일시스템 마운트 명령과 세 번째 동기화가 이어졌다.

세 번째 동기화는 같은 게스트 카운터로 480.684초 후 성공했다. `ATC0_COMMON` 재활성화가 이어졌고, Darwin 이후 1,864.506초에 복원 유형과 `HardwareInfo`를 확인한 결과가 기록됐다. 설치 시작 요청은 보내기 전이었다.

`--verify-on-read`의 복원 저장소는 기록된 64 KiB 블록 해시를 `stela_crypto::Sha256`으로 대조한다. 같은 라이브러리를 연결한 x86-64 UEFI 시험에서 `sha-ni`만 켜고 끈 두 QEMU 실행 모두 64 KiB 블록 16개의 해시가 일치했다. CPUID에 SHA가 있을 때 EFI `.text`의 TCG 추적에서 `sha256rnds2` 등 SHA 명령 위치 56개를 확인했다. SHA를 끈 실행은 같은 EFI로 소프트웨어 경로를 사용했다. 이 시험은 해시 분기와 결과를 확인한다. 설치 시간과 실기 CPUID는 15IGL7에서 측정한다. [복원 블록 검증](../../crates/stela-runtime/src/restore/store.rs)

`native-profile`의 `SPI NOR PROFILE`은 기존 heartbeat마다 직전 표본 이후의 성공한 레지스터·FIFO 접근 횟수와 완료된 NOR 명령의 바이트를 기록한다. `boot-log --perf --text --csv`는 게스트 카운터를 16진수로 보존하고, 접근·명령 횟수와 데이터량을 각각 `count`와 `bytes`로 출력한다. x86 SPI·NOR 시험 19개, x86-64 UEFI Rust 타입 검사와 CLI 시험 9개가 통과했다. 재빌드한 실제 `stela` 명령으로 모델 직렬 로그를 읽고 CSV 출력을 대조했다. 이 계측을 켠 복원 실행으로 실제 PIO 작업량을 대조해야 한다.

## PERF-04 커널의 BMI 대체 경로

동일한 EFI `2ca3df16f87428999aee938b3835b14dfbe190b76fce273972dce9faf367985a`와 macOS 26B123, 4 GiB QEMU에서 `--profile`을 켰다. TCG의 카운트 명령 CPUID 판정을 수정한 QEMU 10.1.3에서 ABM을 켜고 AVX·BMI1·BMI2를 끈 실행은 SSE4.2·SHA·LZCNT 번역 대상으로 Darwin에 진입했다. AVX·BMI1·BMI2를 켠 대조 실행도 Darwin에 진입했다. Intel의 N4120 사양은 SSE4.2를 명시한다. 15IGL7의 실제 CPUID 및 처리율은 실기 실행에서 수집해야 한다.  ·  · [Intel N4120 사양](https://www.intel.com/content/www/us/en/products/sku/197309/intel-celeron-processor-n4120-4m-cache-up-to-2-60-ghz/specifications.html)

`--budget`으로 정지한 BMI 비활성 실행에서 `VBAR_EL2`의 64바이트를 덤프했다. 덤프는 같은 커널캐시의 `__TEXT_EXEC`에서 정확히 한 곳과 일치했다. 초기 벡터 전환을 포함한 첫 구간을 제외하고, 뒤의 7개 `PES HOT kernel` 구간에서 3,256개 표본을 얻었다. 로그에 표시된 상위 블록 중 일치한 64개의 실제 ARM64 바이트를 같은 시스템 모드 프로필과 SSE4.2+LZCNT 목표에서 BMI1/2 유무만 바꿔 다시 컴파일했다. 해당 64개 블록의 목록상 표본은 총 1,864개다. 64개 중 5개의 길이 차이는 모두 `BIC W16,W16,W15`였다. 이번 표본에서 가변 시프트는 관측되지 않았다.

반전 피연산자가 목적지와 다른 호스트 레지스터에 이미 있을 때 `(left | right) ^ right`로 계산한다. 해당 행은 BMI 없는 대상에서 `MOV/NOT/AND` 8바이트에서 `OR/XOR` 6바이트로 줄었다. 선택한 64개 블록의 합계는 13,841에서 13,831바이트로 줄었고, BMI 대상의 `ANDN`과 13,826바이트 합계는 유지됐다. 수치는 블록 재컴파일 결과다. x86_64 macOS에서 순차 실행한 pinned 레지스터 테스트 14개가 통과했다. [BIC 명령 선택](../../crates/stela-compiler/src/x86_64/pinned.rs)

CPU 기능이 없을 때의 다른 경로는 가변 시프트의 CL 시프트, CTZ의 BSF와 영 입력 처리, fast PDEP 전용 융합 건너뛰기, 16바이트 원자 읽기가 필요한 경우 컴파일러 요구 사항 검사다. 기능 마스크를 그대로 N4120의 실측 CPUID로 해석하면 안 된다. [시프트](../../crates/stela-compiler/src/x86_64/integer_math.rs) · [CTZ](../../crates/stela-compiler/src/x86_64/integer_unary.rs) · [PDEP 융합](../../crates/stela-compiler/src/x86_64/block.rs) · [원자 읽기](../../crates/stela-compiler/src/x86_64/ordered.rs)

## 실기 측정표

| 시나리오와 조건 | 지표 | 기준 빌드 | 후보 빌드 | 로그 또는 관찰 위치 |
| --- | --- | --- | --- | --- |
| S4, 캐시가 비어 있는 첫 부팅 | 데스크톱 준비 시간, `compiled` 합계 | | | |
| S4, 정상 종료 뒤 두 번째 부팅 | 데스크톱 준비 시간, `compiled` 합계 | | | |
| S5, 동일한 5분 구간 | 처리율, `HOST SERVICE` 표시 비용, 스왑 I/O | | | |
| S6, 동일한 페이지 | 입력 가능 시간, 표시 비용, 화면 깨짐 관찰 | | | |
