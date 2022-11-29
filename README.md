# A richer `Result` type for Rust

Defined as
```rs
pub enum Result<T, RE, FE> {
    Ok(T),
    Recoverable(RE),
    Fatal(FE),
}
```
it can handle recoverable & fatal errors somewhat easily.

Using `?` on it, it either diverges with `Result::Fatal` for a `Result::Fatal`, or returns a
```rs
pub enum LocalResult<T, RE> {
    NoErr(T),
    Handle(RE),
}
```
which in turn can be used with `?` to get the `T` out of it, or diverge with a `Result::Recoverable`.

`Result` from `core` when used with `?` either diverges with `Fatal` or returns the value in `Ok`.

Additionally, for a public API surface for example, you can use the `Result` type from `core` by
stacking it like so: `Result<Result<T, RE>, FE>`, with all the instances of `?` working as expected.
