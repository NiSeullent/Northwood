use super::*;

#[test]
fn every_case_is_recorded() {
    let table = recorded().unwrap();
    let cases = corpus::all();
    assert_eq!(table.len(), cases.len());
    for case in &cases {
        assert!(table.contains_key(case.name.as_str()), "{}", case.name);
    }
}

/// The regression check: every case through Stela gives the outcome
/// recorded on the Mac, except the known differences, which still differ.
#[test]
fn stela_matches_the_recorded_mac() {
    let table = recorded().unwrap();
    let cases = corpus::all();
    let endings = stela::run(&cases, TargetCpu::detect()).unwrap();
    let mut unexpected = Vec::new();
    for (case, ending) in cases.iter().zip(&endings) {
        let stela = record::outcome(case, ending);
        let mac = table[case.name.as_str()];
        match known_difference(&case.name) {
            None if stela != mac => {
                unexpected.push(format!("{}: mac {mac}, stela {stela}", case.name))
            }
            Some(_) if stela == mac => unexpected.push(format!("{}: no longer differs", case.name)),
            _ => {}
        }
    }
    assert!(unexpected.is_empty(), "{}", unexpected.join("\n"));
}
