# Northbridge CPU Vulkan 조사 (DRV-05)

## 호스트에서 확인한 범위

2026-11-11에 x86-64 AlmaLinux 10.2에서 컴퓨트 셰이더를 실행했다. 테스트는 Vulkan 큐에 4개 워크그룹을 제출하고 호스트가 매핑한 버퍼에서 64개 값을 모두 검사한다. Mesa lavapipe와 Android Emulator의 SwiftShader가 같은 결과를 냈다. 테스트는 공개 Vulkan API와 SPIR-V를 사용한다. 게스트의 AIR→NIR 경로, 3D 합성, UEFI 실행, 15IGL7 성능은 별도 검증 대상이다.

| 호스트 조건 | 정수 컴퓨트 | Stela Metal이 요구하는 기능 그룹 중 빠진 항목 |
| --- | --- | --- |
| lavapipe, Mesa 25.2.7, LLVM 21.1.8, Vulkan 1.4.318 | 통과, 64개, 체크섬 86432 | `unifiedImageLayouts` |
| 같은 lavapipe, 128비트 벡터, 스레드 2개 | 통과, 64개, 체크섬 86432 | `unifiedImageLayouts` |
| 같은 lavapipe, Gallium CPU 기능을 SSE4.1로 제한, 128비트 벡터, 스레드 2개 | 통과, 64개, 체크섬 86432 | `unifiedImageLayouts`, `shaderFloat16` |
| SwiftShader Device (Subzero), Vulkan 1.3.0 | 통과, 64개, 체크섬 86432 | `unifiedImageLayouts`, `shaderFloat16`, `storageImageWithoutFormat`, `maintenance5`, `nullDescriptor`, `conditionalRendering`, `dynamicRenderingUnusedAttachments`, `dynamicPrimitiveTopologyUnrestricted` |

두 드라이버 모두 PCI 장치 식별 기능을 보고하지 않았다. lavapipe는 DMA-BUF 관련 확장 두 개를 보고했고, SwiftShader는 그 조합을 보고하지 않았다. 두 드라이버 모두 `VK_EXT_external_memory_host`를 보고했다. 실제 게스트 공유 메모리 가져오기는 시험하지 않았다. Gallium 옵션은 해당 드라이버가 감지하는 CPU 기능을 제한한다. 15IGL7의 CPUID, 메모리 예산 및 장치 성능은 해당 장비에서 측정한다. [Mesa LLVMpipe 옵션](https://docs.mesa3d.org/drivers/llvmpipe.html) · [Mesa Gallium CPU 기능 옵션](https://docs.mesa3d.org/envvars.html)

2026-11-12 재시험에서 Gallium의 CPU 로그는 SSE4.2 사용, AVX·AVX2·F16C 비활성화를 표시했다. `shaderFloat16`을 요청한 `vkCreateDevice`는 `VK_ERROR_FEATURE_NOT_PRESENT`를 반환했다. 정수 셰이더와 `PackHalf2x16` 변환은 각각 64개 결과가 일치했다. 변환 시험에는 Float16 산술이 포함되지 않는다.

고정 Mesa 26.2.5의 소스에는 `unifiedImageLayouts`가 활성화돼 있다. x86의 Float16 기능은 Gallivm의 F16C 판정에 의존하고, Mesa는 사용 가능한 AVX가 없으면 F16C를 끈다. [lavapipe 기능 선언](https://gitlab.freedesktop.org/mesa/mesa/-/blob/ddaf85e353442d9d7f192fdcf1823e652e3f2ee3/src/gallium/frontends/lavapipe/lvp_device.c) · [Gallivm 기능 판정](https://gitlab.freedesktop.org/mesa/mesa/-/blob/ddaf85e353442d9d7f192fdcf1823e652e3f2ee3/src/gallium/auxiliary/gallivm/lp_bld_limits.h#L81)

고정 Mesa 26.2.5를 AlmaLinux에 별도로 빌드해 Vulkan 1.4.354의 실제 CPU 장치에서 확인했다. AVX·F16C를 끈 경우에도 `unifiedImageLayouts`를 켠 논리 장치에서 정수 컴퓨트와 오프스크린 이미지 지우기가 통과했다. Stela의 `probe_software`는 PCI 주소 없이 lavapipe를 선택하고 저장소 버퍼의 호스트 포인터 가져오기 정렬을 4096바이트로 확인했다. 기존 공통 Metal 기능 검사 20개 중 AVX·F16C 비활성 조건에서 부족한 항목은 `shaderFloat16`이었다. CPU 전용 검사가 일반 AIR에 필요한 FP32 비정규수 보존도 조회했다. 해당 Mesa는 두 CPU 조건 모두 `shaderDenormPreserveFloat32=false`를 보고했다.

같은 AVX·F16C 비활성 Mesa에서 소프트웨어 FP16 `add/sub/mul/fma` 대조 셰이더의 2,097,152개 결과가 정수 참조와 일치했다. 보정을 빼면 `0x57ff × 0x3c01 + 0x0401`의 FMA가 `0x5801` 대신 `0x5800`으로 반올림됐다. `AirCompiler::new_linked_software`는 계산 및 제한된 정점·프래그먼트 셰이더의 변환과 독립 NIR 캐시 키를 선택한다. Linux x86-64에서 LLVM 23.1.6으로 실제 AIR를 컴파일하고 고정 lavapipe의 내부 NIR 경로로 제출했다. FMA 경계값 5개와 오프스크린 RGBA8 픽셀 16개가 기대값과 일치했고, 두 시험 모두 Vulkan에 가져온 원래 호스트 할당의 검사 바이트를 보존했다. 그래픽 시험은 명시적 FP32 flush-to-zero, 32비트 스테이지 입출력, 단일 샘플을 사용한다.

고정 lavapipe의 기본 계산 작업은 FP32 `0x00400000 × 2`를 0으로 처리했다. 계산 작업별 비정규수 모드를 적용한 ICD에서 명시적 보존 작업은 `0x00800000`을 반환했고, 같은 장치에서 번갈아 실행한 기본·명시적 flush 작업은 부호 있는 0을 유지했다. 작업 스레드 0개와 2개를 각각 검증했다. 고정 커밋에 Mesa 패치 27개를 차례로 적용한 결과에서 해당 소스는 시험한 ICD의 빌드 소스와 바이트 단위로 일치했다. 검증 범위는 llvmpipe 내부 계산 작업이며 Vulkan 기능 보고는 기존 값을 반환한다. [Mesa 패치](../third_party/patches/mesa/0027-compute-fp32-denorm-mode.patch)

정점·프래그먼트의 FP32 비정규수 보존을 삼각형과 사각형에서 확인했다. 한 렌더링에서 모드를 번갈아 바꿨고, 부분 스탬프가 생기는 17픽셀 사각형을 포함해 래스터 작업 스레드 0개와 2개 모두에서 모든 픽셀이 기대값과 일치했다. 같은 ICD에서 계산 모드 전환과 실제 AIR `float4`·`half4` 출력도 각각 통과했다. FP32 그래픽 시험은 셰이더 산술을 검증했고, 일반 보간·블렌딩 및 공개 Vulkan 기능 판정은 그대로다. Mesa 패치 29개를 순서대로 적용해 래스터 관련 소스가 시험한 빌드와 바이트 단위로 일치함을 확인했다. [래스터 패치](../third_party/patches/mesa/0028-render-fp32-denorm-mode.patch)

정적 lavapipe 빌드는 NVK와 겹치는 `vk_icd*` 심벌을 제외하고, 직접 장치 핸들로 정점·프래그먼트·컴퓨트 NIR 옵션을 제공한다. Linux 정적 링크에서 세 옵션이 같은 포인터임을 확인했다. AMD64 COFF의 대표 소스 16개도 컴파일했다. 공용 Vulkan 오브젝트 통합과 전체 EFI 링크는 남아 있다. [정적 연결 패치](../third_party/patches/mesa/0029-static-lavapipe-bridge.patch)

AVX·F16C를 끈 같은 lavapipe에서 4096바이트 정렬의 호스트 매핑을 `HOST_ALLOCATION`으로 가져왔다. 셰이더가 Vulkan 버퍼 주소와 64바이트 바인딩 오프셋으로 접근해 원래 매핑의 64개 값을 갱신했고, 양쪽 검사 바이트를 보존했다. 실제 게스트 프레임에 연결할 때는 Vulkan 메모리 해제까지 원래 매핑과 프레임을 유지해야 한다. Vulkan 규약상 메모리 핸들이 살아 있는 동안 동일 물리 메모리의 다른 호스트 주소로 접근할 수 없으므로, 현재 NVK의 별칭 매핑과는 다른 경로가 필요하다.  · [Vulkan 호스트 메모리 규약](https://docs.vulkan.org/spec/latest/chapters/memory.html#memory-external-host)

Rust의 `HostAllocation`은 원래 매핑을 Vulkan 메모리 해제까지 소유하고 버퍼의 외부 메모리 요건을 확인한다. AVX·F16C를 끈 Linux lavapipe에서 Vulkan 전송 명령이 원래 매핑의 오프셋 2048부터 256바이트를 갱신하고 검사 바이트와 매핑 수명을 보존했다. 작업 스레드 0개·2개에서 같은 결과를 얻었다. 장치 초기화는 필수 기능 검사를 유지한다.

외부 메모리 이미지의 초기 레이아웃은 `UNDEFINED`이며 첫 전환에서 기존 픽셀이 유실될 수 있다. 별도 Vulkan 이미지에서 먼저 초기 전환을 마치고, 원래 호스트 버퍼의 BGRA 17×3 픽셀을 올려 읽은 다음 새 픽셀을 버퍼로 돌려보내는 시험을 작업 스레드 0개·2개에서 통과했다. 128바이트 행의 패딩과 검사 바이트는 보존됐다. 외부 선형 이미지의 실제 행 간격도 17·1366·1920픽셀에서 확인했다. 게스트 프레임 연결과 IOSurface 접근 전후에 복사를 예약하는 서비스 경계는 남아 있다.  · [Vulkan 이미지 초기 레이아웃](https://docs.vulkan.org/refpages/latest/refpages/source/VkImageCreateInfo.html)

게스트 kext는 양방향으로 고정한 메모리의 토큰에만 쓰기 가능 비트를 붙인다. 게이트는 토큰 전체를 반환하고 kext는 반환 시 비트를 지워 원래 디스크립터를 해제한다. CPU Vulkan은 이 비트가 없거나 디스크립터 값이 0인 요청을 매핑 전에 거부한다. arm64e kext 컴파일, 게이트 토큰 왕복과 실제 lavapipe의 거부·BGRA 왕복을 확인했다. 게스트 실행과 실제 IOSurface의 픽셀 검증은 남아 있다.

Metal의 버퍼 텍스처 생성 경로는 호스트 포인터를 가져온 공유 BGRA8 렌더 타깃에 별도 Vulkan 이미지를 할당한다. 전체 크기의 단일 샘플 첨부에서 `Load`는 렌더 전에 업로드하고 최종 `Store`는 렌더 후 원래 버퍼로 복사한다. 공유 이벤트보다 먼저 CPU 가시성 장벽을 기록한다. 일반 Device 시험에서 행 간격, 지연 Store, 이벤트 대기 중 수명, NVK의 기존 선형 경로를 확인했다. 이 시험은 모의 Vulkan 백엔드를 썼으며, 실제 lavapipe에서는 별도 BGRA 왕복 시험을 수행했다. [Device 회귀 시험](../crates/stela-driver-gpu/src/metal/device/tests.rs)

`AirCompiler::new_linked_lavapipe`는 고정된 패치 드라이버용 NIR 프로필과 별도 캐시 키를 선택한다. FP32 덧셈·뺄셈·곱셈과 네 비교 연산의 비정규수 보존을 확인했다. FMA와 보간 등 추가 보존 요건이 있는 연산은 이 프로필에서 거부한다.

AVX·F16C를 끈 같은 Linux lavapipe에서 기존 `metal_abi.c`가 만든 내부 NIR 파이프라인 11개를 생성했다. 작업 스레드 0개·2개에서 누산 결과 71과 XR10의 FP16 포장·복원에 사용한 색상값 3개가 원래 호스트 매핑에 일치했다. 전체 Metal 장치 등록과 나머지 내부 파이프라인의 실행은 검증하지 않았다.

선택형 호스트 Metal 서비스는 고정 lavapipe의 전용 NIR ABI로 열고, GPU family 0개와 단일 샘플 기능을 협상한다. 실제 Linux의 서비스 프로토콜에서 `BeginRender`의 BGRA8 Clear, 지연 Store, `Commit`과 완료를 실행해 원래 17×3 호스트 버퍼의 픽셀·행 패딩·보호 바이트와 반환된 양방향 토큰을 확인했다. AVX·F16C 비활성 및 작업 스레드 0개·2개 조건에서 통과했다. 기본 소프트웨어 초기화는 Vulkan의 실제 FP16·FP32 기능을 확인한다. 게스트 부팅은 호스트 전용 서비스의 검증 범위에 포함되지 않았다.

같은 서비스에서 1366×768 BGRA8 전체 프레임을 5,632바이트 행 간격과 128바이트 오프셋으로 시험했다. Clear와 지연 Store 뒤, CPU가 쓴 새 패턴을 Load와 Store로 왕복시켰다. 두 패스 모두 후속 작업이 대기 중일 때 공유 이벤트가 신호를 보내면 픽셀과 행 패딩, 바깥 보호 영역이 일치했다. 두 작업 스레드 설정에서 양방향 토큰의 반환도 일치했다.

`WiredRam`은 게스트 고정 메모리의 순서와 등록 소유자를 보관한다. 논리 오프셋으로 첫 구간을 찾아 별도 호스트 버퍼와 등록 RAM 사이에서 읽기와 게시를 수행한다. 등록 RAM 시험에서 물리 순서가 뒤집힌 구간, 잘못된 범위의 무효과, 다음 구간이 닫힌 경우 정확한 실패 위치를 확인했다. `GuestRamCopy`는 이를 GPU의 양방향 복사 임대로 제공한다. 등록 소유자 수명과 복사 실패 시 장치 손실 반환을 x86에서 확인했고 UEFI 타입 검사도 통과했다.

복사 경로는 전용 Vulkan 버퍼에 그린 뒤 완료 이벤트 전에 출력 행을 등록 RAM에 게시한다. 실제 Linux lavapipe와 시험용 RAM 제공자로 1366×768 프레임의 Clear·지연 Store, CPU 쓰기 뒤 Load·Store, 행마다 복사되는 범위 및 토큰 반환을 확인했다. 게스트 부팅 경로의 장치 개설과 화면 표시는 남아 있다.

같은 프레임 시험을 실제 `GuestRamCopy`로 반복했다. 물리 주소 순서가 뒤집힌 두 등록 RAM 구간에서 모든 픽셀·행 패딩·보호 영역이 일치했고, 이벤트 가시성과 태그가 붙은 토큰 반환도 확인했다. Linux 호스트의 제출부터 이벤트까지 개발 빌드에서 318~329 ms, `stela-engine`에만 `opt-level=2`를 적용하면 75~77 ms였다. 부팅 쪽에서는 등록 RAM 확보 뒤 장치 트리를 만들기 전에 서비스 배치를 결정하고, 완료 FD나 매핑된 마커가 없는 경로를 부팅 CPU가 보드 잠금 밖에서 진행시켜야 한다.  ·  · [부팅 연결 위치](../crates/stela-boot/src/engine/start.rs) · [부팅 CPU 처리 루프](../crates/stela-boot/src/engine/run.rs)

SoC 보드 모델에서 늦게 등록한 GPU 게이트가 외부 완료를 받아 AIC 인터럽트 700을 올렸고, 게스트가 게이트의 ACK 레지스터에 쓰자 신호가 해제됐다. 완료 FD와 매핑된 마커 없이 시험했다. 등록 RAM의 단일 예약 키 탐색 최적화도 같은 프레임으로 대조했으며 측정 범위가 겹쳐 기존 모니터 구현을 유지했다.

등록 RAM 변경을 포함한 격리 Linux x86-64 엔진 빌드에서 메모리 시험 502개가 통과했다. 3개는 제외했다. 같은 범위의 macOS Rosetta 대조는 네이티브 메모리 레시피와 관련한 실패를 포함하므로 Linux 결과와 각각 기록했다.

`metal-linked`의 `open_copied_lavapipe`를 호스트와 x86-64 UEFI 라이브러리에서 공유한다. 기존 호스트 직접 할당·복사 호출의 타입 검사와 UEFI 라이브러리 검사를 통과했다. `stela-boot-core`의 `GuestRamCopy`도 같은 UEFI 빌드에서 타입 검사를 통과했다. UEFI 호출자가 사용할 ICD와 NIR 옵션은 동일한 정적 Mesa 빌드에서 와야 한다. EFI 링크와 게스트 장치 개설은 남아 있다.

고정 LLVM 23.1.4에서 AMD64 COFF 아카이브 45개와 오브젝트 1,457개를 빌드했다. X86 타겟 등록, SelectionDAG, AsmPrinter, MC와 Stela C·C++ 런타임을 결합한 PE32+ 시험 이미지의 외부 임포트는 0개였다. 별도 호스트 테스트 콜백과 고정 UEFI 타겟을 사용했다. Mesa와 합친 EFI 링크, 펌웨어 부팅과 코드 실행은 남아 있다. [소스 준비 스크립트](../xtask/scripts/llvm-x86-source.sh)

고정 Mesa에 패치 29개를 적용해 lavapipe 프런트엔드 100개와 llvmpipe·Gallivm 지원 오브젝트 607개를 AMD64 COFF로 빌드했다. 고정 LLVM의 ExecutionEngine·MCJIT·RuntimeDyld와 X86Disassembler를 더한 결합 링크에서 미해결 심벌은 35개에서 12개로 줄었다. Mesa 진단 4개, PassBuilder 3개, GDB 리스너 1개, 매핑과 명령 캐시 API 4개가 남아 있다.

후속 결합 링크에 LLVM의 실제 PassBuilder·ProfileData·GDB 등록 구현을 더해 미해결 심벌을 9개로 줄였다. Mesa 스트림 진단 4개, LLVM 신호 등록 1개, 매핑과 명령 캐시 API 4개가 남아 있다.

시험용 COFF 어댑터는 Mesa의 진단 스트림을 LLVM `raw_ostream`으로 연결했다. 고정된 디스어셈블리·IR 덤프와 잘못된 타겟 오류는 호스트에서 원래 출력과 바이트 단위로 일치했다. 결합 링크에서 실제 진단 심벌 4개가 해소돼 미해결 심벌 5개가 남았다.

별도 LLVM 시험 빌드에서 선택형 크래시 IR 계측을 끄자 신호 등록 의존성이 사라졌다. 일반 IR 덤프와 검증 출력은 기본 빌드와 바이트 단위로 일치했다. 결합 링크에서 `llvm::sys::Memory` 심벌 4개가 남았다.

NVK의 EFI용 AIR 소스 목록에 `air_soft_half.c`를 추가했다. 동일 소스의 호스트 오브젝트에서 `air_to_nir.c`가 참조하는 소프트웨어 half 심벌을 대조했다. NVK와 lavapipe를 같은 EFI에 넣는 빌드에는 공용 Vulkan·NIR·C 런타임 아카이브 한 벌이 필요하다. 두 독립 시험 아카이브에는 공용 심벌이 겹친다. lavapipe 옵션 브리지는 선택된 Vulkan 물리 장치에서 vertex `0x01`, fragment `0x10`, compute `0x20`의 NIR 옵션을 조회한다. EFI에서 장치를 열려면 공용 C/C++ 생성자와 PCI GPU에 의존하지 않는 제출·작업 스레드 서비스도 준비돼야 한다.  · [lavapipe 브리지](../third_party/patches/mesa/0029-static-lavapipe-bridge.patch)

lavapipe 소스 준비는 공용 희소 체크아웃 규칙에 ICD 타깃 파일 두 개를 더한다. 고정 소스에서 Meson이 lavapipe 라이브러리와 ICD 타깃을 생성했다. NVK와 lavapipe를 함께 켠 구성은 UEFI Rust 검사와 `bindgen` 확인을 지나 기본 CLC에서 `LLVMSPIRVLib` 23.1 계열, 시스템 CLC 옵션에서 `mesa_clc`를 요구했다. 결합 빌드 그래프는 아직 생성되지 않았다.

고정 Mesa의 `mesa_clc`와 `vtn_bindgen2`를 macOS arm64 도구로 빌드했다. 첫 도구가 OpenCL 함수를 532바이트 SPIR-V로 변환했고, 두 번째 도구가 대응하는 NIR 빌더를 생성했다. 후속 결합 구성은 두 도구를 인식한 뒤 NVK의 `libelf` 의존성에서 멈췄다.

macOS arm64의 실제 `libelf`, 해시를 대조한 Rust 의존 아카이브, 고정 `cbindgen`을 사용해 `--wrap-mode=nodownload`로 두 드라이버의 결합 Ninja 그래프를 생성했다. `ninja -t inputs`에서 NVK 입력 907개와 lavapipe 입력 716개 중 공용 오브젝트 433개를 확인했다. 다음 단계는 공용 오브젝트를 한 번만 빌드하도록 UEFI COFF 타깃에 옮기고 실제 아카이브와 EFI를 링크하는 것이다.

실제 `xtask gpu lavapipe --combined-plan`으로 이 그래프의 입력 1,190개를 NVK 전용 474개, 공용 433개, lavapipe 전용 283개로 분류했다. 두 드라이버가 가리키는 공용 `vk_instance.c.o`의 생성 헤더를 준비한 뒤 기존 리타기팅 규칙으로 AMD64 COFF 오브젝트를 한 번 빌드했다.

패치 `0030`은 lavapipe·llvmpipe의 유한 대기에서 Mesa의 `CLOCK_MONOTONIC` 조건 변수를 사용한다. 고정 소스의 변경된 C 파일 두 개를 Linux에서 빌드해 실시간 시계 요청을 거부하는 조건에서 시간 초과와 신호·무기한 대기를 검증했다. UEFI에서 PCI GPU 없이 기존 Linux 작업 런타임을 연 뒤 실제 제출 스레드의 생성·신호·종료를 검증하는 단계가 남았다.

같은 패치의 대기 파일과 단조 조건 변수 구현을 AMD64 COFF로 빌드했다. 객체의 대기 참조를 기존 Stela pthread·시계 구현의 심벌과 대조하고 단조 시계를 지정하는 인자를 확인했다. 결합 그래프의 공용 Vulkan 동기화·큐 오브젝트 다섯 개는 별도 부분 아카이브로 묶었다. 전체 드라이버와 EFI 링크는 남아 있다.

## 재현

Linux x86-64, Vulkan 헤더와 로더, `glslc`, lavapipe ICD가 설치된 환경에서 저장소 루트 기준으로 실행한다. `lvp_icd`는 배포판이 설치한 ICD JSON 경로로 지정한다.

```sh
work=$(mktemp -d)
lvp_icd=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
glslc xtask/scripts/northbridge-smoke.comp -o "$work/smoke.spv"
cc -std=c11 -Wall -Wextra -Werror -O2 xtask/scripts/northbridge-smoke.c \
  -l:libvulkan.so.1 -o "$work/northbridge-smoke"
VK_DRIVER_FILES="$lvp_icd" "$work/northbridge-smoke" "$work/smoke.spv"
VK_DRIVER_FILES="$lvp_icd" GALLIUM_OVERRIDE_CPU_CAPS=sse4.1 \
  LP_NATIVE_VECTOR_WIDTH=128 LP_NUM_THREADS=2 "$work/northbridge-smoke" "$work/smoke.spv"
```

SwiftShader에서는 `VK_DRIVER_FILES`에 해당 ICD JSON을 지정하고 실행 인자 끝에 `--swiftshader`를 붙인다. 헤더가 `VK_KHR_unified_image_layouts`의 구조체보다 오래된 경우, 확장 목록에도 없으면 `0`, 확장 목록에 있으면 `unknown`이 출력된다. 위 실행은 기존 Android NDK에서 `vulkan`, `vk_video` 헤더만 Linux 컴파일러의 추가 탐색 경로로 사용했다. [Khronos 컴퓨트 결과 읽기 동기화 예제](https://docs.vulkan.org/guide/latest/synchronization_examples.html)

`cargo xtask gpu lavapipe`는 Linux x86-64에서 고정 Mesa와 패치로 ICD를 빌드하고, 출력한 `lavapipe.json`에 소스와 ICD 해시를 기록한다. Meson·Ninja, LLVM 개발 파일과 `glslangValidator`가 필요하다. Linux에서 Vulkan 로더 개발 파일 및 LLVM 23과 함께 빌드한 `stela_metal_compiler_test`는 해당 ICD를 `VK_DRIVER_FILES`로 지정한 뒤 `--lavapipe-software-half`와 `--lavapipe-software-render`로 실행한다. AVX 지원 호스트에서는 `GALLIUM_OVERRIDE_CPU_CAPS=sse4.1 LP_NATIVE_VECTOR_WIDTH=128`을 사용한다.

## 부팅 경로에 연결하기 전에 필요한 작업

1. **AVX 없는 CPU의 셰이더 지원.** 소프트웨어 Float16은 계산과 제한된 정점·프래그먼트 산술, 기본 `half4` 색상 출력을 호스팅 경로에서 검증했다. FP32 보존 시험을 일반 셰이더와 렌더링 연산으로 넓히고 좁은 메모리·스테이지 입력을 지원한 뒤 Metal 기능 조건에 연결한다. [Mesa 소프트웨어 Vulkan 빌드 옵션](https://docs.mesa3d.org/install.html)
2. **CPU 장치 개설.** `probe_software`는 lavapipe CPU 장치와 공용 큐를 찾고 가져오기 요구 사항을 확인한다. `NativeBackend`와 Metal 서비스는 호스트 버퍼를 쓸 수 있으며, 게스트 부팅 연결은 별칭을 피하는 메모리 경로와 완료 신호의 순서를 함께 검증해야 한다. Stela의 x86 섀도 페이지와 물리 메모리 창은 같은 게스트 프레임에 서로 다른 호스트 주소를 사용한다. 이 프레임을 표준 호스트 포인터 가져오기에 직접 연결하면 Vulkan 별칭 제한을 만족할 수 없다. 별도 GPU 버퍼를 사용할 경우 `RegisteredSystemRam`의 기존 정합 읽기·게시 경로를 통해 게스트 데이터를 전달하고, 렌더링 쓰기 결과가 게스트 이벤트보다 먼저 게시됐는지 확인해야 한다. [Vulkan 호스트 메모리 규약](https://docs.vulkan.org/refpages/latest/refpages/source/VkImportMemoryHostPointerInfoEXT.html) · [게스트 RAM 전송](../crates/stela-engine/src/memory/system_ram/transfer.rs)
3. **프리스탠딩 빌드와 메모리.** `xtask/src/gpu/nvk.rs`의 GPU 빌드는 Nouveau 전용이며 Gallium과 LLVM을 끈다. `xtask/src/gpu/llvm.rs`의 LLVM 아카이브는 AIR 처리 위주이고 `llvm.sparse`는 `llvm/lib/Target/*/`를 제외한다. X86 코드 생성과 Gallivm JIT에 필요한 단위를 별도로 고정하고, `compat/libc`의 스레드 생성과 할당을 활용하되 저메모리 프로필에서 워커 수, 스택, 타일, JIT 캐시 예산을 검증한다. `mmap`은 `PROT_EXEC`를 거부하고 `mprotect`는 지원하지 않는 페이지 권한 전환에 `ENOTSUP`을 반환한다. JIT는 실행용 페이지의 쓰기·실행 전환과 동기화, 해제를 구현해야 한다.
4. **통합과 M4 판정.** 오프스크린 NIR 컴퓨트와 그래픽 결과를 확인한 뒤 CPU 이미지 출력을 `GopScanout`에 연결한다. CPU 렌더러 선택, 초기화 실패, 비활성화 상태의 GOP 부팅을 QEMU에서 검증한다. 15IGL7에서 설치를 마치고 WindowServer가 Metal 장치로 데스크톱을 합성하는지 확인한다. MEM-05와 함께 힙·페이징 압박과 S4–S6 성능을 기록한 뒤 기본 동작을 정한다.
