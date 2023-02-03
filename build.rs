use phf_codegen::Map;

fn main() {
    let mut map = Map::new();
    let data = include_str!("./codata.txt");
    // based off of parse_constants_2018toXXXX from scipy.constants.codata
    for line in data.lines().skip(11) {
        let name = line[..60].trim_end();
        let val = line[60..85].trim_end().replace(' ', "_").replace("...", "");
        let uncert = line[85..110]
            .trim_end()
            .replace(' ', "_")
            .replace("(exact)", "0");
        let units = line[110..].trim_end();
        map.entry(name, &format!("({}_f64, {}_f64, {:?})", val, uncert, units));
    }
    let mut path = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    path.push("codata.rs");
    std::fs::write(path, map.build().to_string()).unwrap();
}
