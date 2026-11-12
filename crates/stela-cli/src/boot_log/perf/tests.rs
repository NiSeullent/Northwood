// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;

const SOURCE: &str = "serial,\"file\".txt";

fn report(input: &[u8], overwritten: bool, csv: bool) -> String {
    let mut output = Vec::new();
    if csv {
        csv_header(&mut output).unwrap();
    }
    write_report(input, &mut output, SOURCE, overwritten, csv).unwrap();
    String::from_utf8(output).unwrap()
}

#[test]
fn qemu_serial_capture_uses_actual_memory_and_service_fields() {
    // These lines were extracted from the QEMU TCG Stela 0.1.0 smoke run.
    // The guest stopped before it emitted a heartbeat.
    let input = include_bytes!("qemu-smoke.txt");
    let summary = report(input, false, false);
    assert!(
        summary
            .contains("1 | - | compact | 3672/2244 | - | - | - | - | 31591 | - | samples present"),
        "{summary}"
    );
    let csv = report(input, false, true);
    for row in [
        "HOST MEMORY,,heap,1298,MiB",
        "HOST MEMORY,,profile,compact,text",
        "HOST SERVICE,,maximum_gap_us,31591,us",
        "HOST SERVICE,gap,kind,guest,text",
        "HOST SERVICE,longest,display,96,us",
        "HOST SERVICE,total,display,145,us",
        "HOST SERVICE,driver_longest,log,187,us",
        "HOST SERVICE,driver_total,log,204,us",
        "HOST SERVICE,drivers,host_drivers_held_us,88221,us",
    ] {
        assert!(
            csv.contains(&format!("\",1,,3,{row}\n")) || csv.contains(&format!("\",1,,2,{row}\n")),
            "{row}: {csv}"
        );
    }
    assert!(csv.contains("\"serial,\"\"file\"\".txt\",1,,2,HOST MEMORY"));
}

#[test]
fn accepts_the_host_memory_fallback_diagnostic() {
    let input = b"HOST MEMORY free=2463M before_reclaim=2101M cores=4\n\
        HOST MEMORY heap_contiguous_unavailable=593M fallback=536M\n\
        HOST MEMORY free=2463M heap=536M code=64M pool=64M guest=1799M profile=paged\n";
    let csv = report(input, false, true);
    assert!(csv.contains(",1,,2,HOST MEMORY,,heap_contiguous_unavailable,593,MiB\n"));
    assert!(csv.contains(",1,,2,HOST MEMORY,,fallback,536,MiB\n"));
    assert!(
        report(input, false, false)
            .contains("1 | - | paged | 2463/1799 | - | - | - | - | - | - | samples present")
    );
}

#[test]
fn extracts_spi_nor_counts_and_bytes_at_the_guest_counter() {
    let input = b"=== STELA BOOT 3 ===\n\
        [   8.125] SPI NOR PROFILE counter=0x201 mmio_reads=6 mmio_writes=12 \
        rxdata_reads=4 txdata_writes=4 reads=1 read_bytes=4 programs=0 program_bytes=0 \
        erases=0 erase_bytes=0 status_bytes=1\n\
        SPI NOR PROFILE counter=0x301 mmio_reads=0 read_bytes=0\n\
        SPI NOR PROFILE counter=invalid read_bytes=12345\n";
    let csv = report(input, false, true);
    for row in [
        ",1,3,2,SPI NOR PROFILE,,timestamp,8.125,s",
        ",1,3,2,SPI NOR PROFILE,,counter,0x201,hex",
        ",1,3,2,SPI NOR PROFILE,,rxdata_reads,4,count",
        ",1,3,2,SPI NOR PROFILE,,read_bytes,4,bytes",
        ",1,3,2,SPI NOR PROFILE,,status_bytes,1,bytes",
        ",1,3,3,SPI NOR PROFILE,,mmio_reads,0,count",
        ",1,3,3,SPI NOR PROFILE,,read_bytes,0,bytes",
    ] {
        assert!(csv.contains(&format!("{row}\n")), "{row}: {csv}");
    }
    assert!(!csv.contains("12345"));
    assert!(report(input, false, false).contains("| partial\n"));
}

#[test]
fn reports_interval_totals_separately_when_boot_markers_repeat() {
    let input = b"=== STELA BOOT 2 ===\r\n\
        STELA UEFI MOUNT version=0.1.0\r\n\
        HOST MEMORY free=2700M cores=4 staged=0x100 loader_released=0x400\r\n\
        HOST MEMORY free=2700M heap=536M code=64M pool=64M guest=1950M profile=paged\r\n\
        [   3.057] BOOT STAGE name=guest-dispatch elapsed_ms=3057\r\n\
        BEAT blocks=10 ticks/block=9 native=62% compiled=2 discards=7(recycle=3,eviction=4) heap=10M/20M shadow_peak=1M/2M/8M code_peak=5M/64M\r\n\
        GUEST PERFORMANCE retired=100 interval_retired=100 instructions_per_second=10 instructions_per_block_milli=10000 loops=2/3/4\r\n\
        ENGINE MEMORY pe=0 blocks=2M owners=3M successors=5M rows=7M index=11M lookup=13M pages=17M\r\n\
        BEAT COST us position=17 service=2 steps=4 longest_step=11\r\n\
        HOST CLOCK cpu=bsp aperf_delta=180 mperf_delta=100 ratio_milli=1800 active_mhz=1980\r\n\
        BEAT blocks=50 ticks/block=7 native=72% compiled=3 discards=11\r\n\
        GUEST PERFORMANCE retired=500 interval_retired=400 instructions_per_second=40 instructions_per_block_milli=10000 loops=2/3/4\r\n\
        === STELA BOOT 2 ===\r\n\
        STELA UEFI MOUNT version=0.1.0\r\n\
        BEAT blocks=12 ticks/block=4 native=1% compiled=0 discards=0\r\n";
    let summary = report(input, false, false);
    assert!(
        summary.contains("1 | 2 | paged | 2700/1950 | 2 | 5 | 25 | 58 | - | 11 | samples present"),
        "{summary}"
    );
    assert!(
        summary.contains("2 | 2 | - | -/- | 1 | 0 | - | - | - | - | samples present"),
        "{summary}"
    );

    let csv = report(input, false, true);
    for row in [
        ",1,2,3,HOST MEMORY,,staged,0x100,hex",
        ",1,2,5,BOOT STAGE,,timestamp,3.057,s",
        ",1,2,5,BOOT STAGE,,name,guest-dispatch,text",
        ",1,2,5,BOOT STAGE,,elapsed_ms,3057,ms",
        ",1,2,6,BEAT,,discards,7,count",
        ",1,2,6,BEAT,discard_cause,recycle,3,count",
        ",1,2,6,BEAT,discard_cause,eviction,4,count",
        ",1,2,6,BEAT,heap,total,20,MiB",
        ",1,2,6,BEAT,shadow,pool,8,MiB",
        ",1,2,6,BEAT,,native,62,percent",
        ",1,2,7,GUEST PERFORMANCE,loops,extra_retired,4,count",
        ",1,2,8,ENGINE MEMORY,,index,11,MiB",
        ",1,2,9,BEAT COST,,position,17,us",
        ",1,2,10,HOST CLOCK,,cpu,bsp,text",
        ",1,2,10,HOST CLOCK,,ratio_milli,1800,milli-ratio",
        ",1,2,10,HOST CLOCK,,active_mhz,1980,MHz",
        ",2,2,15,BEAT,,compiled,0,count",
    ] {
        assert!(csv.contains(&format!("{row}\n")), "{row}: {csv}");
    }
}

#[test]
fn extracts_shared_code_window_metrics_per_interval() {
    let input = b"PES CODE live/slots/pages/releases 0:9982/9982/258/0 \
        board=64M/32/1 churn=new/ring/ivau/slot/other:9974/0/8/0/0 \
        lock=acquired/contended/waited_us/longest_us:22318/0/0/0\n\
        PES CODE live/slots/pages/releases 0:10253/10253/284/0 \
        board=64M/32/3 churn=new/ring/ivau/slot/other:10253/0/0/0/0 \
        lock=acquired/contended/waited_us/longest_us:22373/2/108/71\n\
        PES CODE live/slots/pages/releases 0:8/8/2/0\n";
    let csv = report(input, false, true);
    for row in [
        ",1,,1,PES CODE,board,window,64,MiB",
        ",1,,1,PES CODE,board,parts,32,count",
        ",1,,1,PES CODE,board,generation,1,count",
        ",1,,1,PES CODE,churn,ivau,8,count",
        ",1,,2,PES CODE,board,generation,3,count",
        ",1,,2,PES CODE,lock,waited,108,us",
        ",1,,2,PES CODE,lock,longest,71,us",
    ] {
        assert!(csv.contains(&format!("{row}\n")), "{row}: {csv}");
    }
    assert!(!csv.contains(",1,,3,PES CODE"));
    assert!(!report(input, false, false).contains("partial"));
}

#[test]
fn malformed_code_window_tuples_are_never_partially_exported() {
    let input = b"PES CODE live/slots/pages/releases 0:1/1/1/0 board=64M/32\n\
        PES CODE live/slots/pages/releases 0:2/2/2/0 board=64M/32/2 \
        churn=new/ring/ivau/slot/other:1/2/3/4/5/6 \
        lock=acquired/contended/waited_us/longest_us:0/0/unknown/0\n";
    let csv = report(input, false, true);
    assert_eq!(csv.lines().count(), 4, "{csv}");
    assert!(csv.contains(",1,,2,PES CODE,board,generation,2,count\n"));
    assert!(report(input, false, false).contains("partial"));
}

#[test]
fn wrapped_prefix_and_incomplete_lines_do_not_make_false_samples() {
    let input = b"BEAT blocks=5 compiled=999\n\
        BEAT blocks=10 compiled=2\n\
        [boot log dropped 12 bytes]\n\
        === STELA BOOT 9 ===\n\
        HOST MEMORY free=2500M cores=4\n\
        BEAT blocks=20 compiled=999";
    let summary = report(input, true, false);
    assert!(
        summary.contains("1 | - | - | -/- | 1 | 2 | - | - | - | - | partial"),
        "{summary}"
    );
    assert!(
        summary.contains("2 | 9 | - | 2500/- | - | - | - | - | - | - | partial"),
        "{summary}"
    );
    assert!(!report(input, true, true).contains("999"));
}

#[test]
fn serial_restart_and_empty_capture_are_distinct_from_zero_metrics() {
    let input = b"STELA UEFI MOUNT version=0.1.0\n\
        STELA UEFI MOUNT version=0.1.1\n\
        BEAT blocks=1 compiled=0\n";
    let summary = report(input, false, false);
    assert!(
        summary.contains("1 | - | - | -/- | - | - | - | - | - | - | no samples"),
        "{summary}"
    );
    assert!(
        summary.contains("2 | - | - | -/- | 1 | 0 | - | - | - | - | samples present"),
        "{summary}"
    );
    assert_eq!(report(b"", false, true).lines().count(), 1);
}

#[test]
fn drains_oversized_and_non_utf8_lines_before_later_samples() {
    let mut input = vec![b'a'; 100 << 10];
    input.extend_from_slice(b"\n\xff\nHOST MEMORY free=2400M cores=2\n");
    let mut output = Vec::new();
    write_report(
        io::BufReader::with_capacity(32, input.as_slice()),
        &mut output,
        SOURCE,
        false,
        false,
    )
    .unwrap();
    let output = String::from_utf8(output).unwrap();
    assert!(
        output.contains("1 | - | - | 2400/- | - | - | - | - | - | - | partial"),
        "{output}"
    );
}
