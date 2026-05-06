fn main() {
    if std::env::var("PROFILE").as_deref() == Ok("release") {
        static_vcruntime::metabuild();
    }
}
