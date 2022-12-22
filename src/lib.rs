#![feature(try_trait_v2)]
#![cfg_attr(not(any(test, feature = "std")), no_std)]

use core::{convert::Infallible, fmt, iter, ops};

pub use core::result::Result::{self as StdResult, Err as StdErr, Ok as StdOk};

mod sealed {
    use super::StdResult;

    pub trait StdResultExt {
        type Ok;
        type Err;
    }

    impl<O, E> StdResultExt for StdResult<O, E> {
        type Ok = O;
        type Err = E;
    }
}

pub trait StdResultExt: Into<StdResult<Self::Ok, Self::Err>> + sealed::StdResultExt {
    fn rich(
        self,
    ) -> Result<
        <Self::Ok as sealed::StdResultExt>::Ok,
        <Self::Ok as sealed::StdResultExt>::Err,
        Self::Err,
    >
    where
        Self::Ok: StdResultExt,
    {
        Result::from_std(self.into().map(Into::into))
    }

    fn ok_or_recoverable<FE>(self) -> Result<Self::Ok, Self::Err, FE> {
        Result::ok_or_recoverable(self.into())
    }

    fn ok_or_fatal<RE>(self) -> Result<Self::Ok, RE, Self::Err> {
        Result::ok_or_fatal(self.into())
    }

    fn recoverable_or_fatal<T>(self) -> Result<T, Self::Ok, Self::Err> {
        Result::from_err(self.into())
    }

    fn local(self) -> LocalResult<Self::Ok, Self::Err> {
        LocalResult::from_std(self.into())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct CollectedErrs<C>(pub C);

impl<T, E, C, EC> FromIterator<StdResult<T, E>> for CollectedErrs<StdResult<C, EC>>
where
    C: FromIterator<T>,
    EC: FromIterator<E>,
{
    fn from_iter<I: IntoIterator<Item = StdResult<T, E>>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        let first_err = match (&mut iter).collect() {
            StdOk(all_ok) => return CollectedErrs(StdOk(all_ok)),
            StdErr(err) => err,
        };
        CollectedErrs(StdErr(
            iter::once(first_err)
                .chain(iter.filter_map(|i| i.err()))
                .collect(),
        ))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct CollectedRecoverables<T>(pub T);

struct CollectSum<T>(T);

impl<T, U> FromIterator<U> for CollectSum<T>
where
    T: iter::Sum<U>,
{
    fn from_iter<I: IntoIterator<Item = U>>(iter: I) -> Self {
        CollectSum(iter.into_iter().sum())
    }
}

struct CollectProduct<T>(T);

impl<T, U> FromIterator<U> for CollectProduct<T>
where
    T: iter::Product<U>,
{
    fn from_iter<I: IntoIterator<Item = U>>(iter: I) -> Self {
        CollectProduct(iter.into_iter().product())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[must_use]
pub enum Result<T, RE, FE> {
    Ok(T),
    Recoverable(RE),
    Fatal(FE),
}

pub use crate::Result::*;

impl<T, RE, FE> Result<T, RE, FE> {
    pub fn from_std(value: StdResult<StdResult<T, RE>, FE>) -> Self {
        match value {
            StdOk(StdOk(ok)) => Ok(ok),
            StdOk(StdErr(err)) => Recoverable(err),
            StdErr(err) => Fatal(err),
        }
    }

    pub fn from_split<E>(
        split: impl FnOnce(E) -> StdResult<RE, FE>,
        value: StdResult<T, E>,
    ) -> Self {
        match value {
            StdOk(ok) => Ok(ok),
            StdErr(err) => Self::from_err(split(err)),
        }
    }

    pub fn ok_or_recoverable(value: StdResult<T, RE>) -> Self {
        StdOk(value).into()
    }

    pub fn ok_or_fatal(value: StdResult<T, FE>) -> Self {
        value.map(StdOk).into()
    }

    pub fn from_err(error: StdResult<RE, FE>) -> Self {
        match error {
            StdOk(err) => Recoverable(err),
            StdErr(err) => Fatal(err),
        }
    }

    pub fn to_std(self) -> StdResult<StdResult<T, RE>, FE> {
        match self {
            Ok(ok) => StdOk(StdOk(ok)),
            Recoverable(err) => StdOk(StdErr(err)),
            Fatal(err) => StdErr(err),
        }
    }

    pub fn to_std_flipped(self) -> StdResult<T, StdResult<RE, FE>> {
        match self {
            Ok(ok) => StdOk(ok),
            Recoverable(err) => StdErr(StdOk(err)),
            Fatal(err) => StdErr(StdErr(err)),
        }
    }

    pub const fn is_ok(&self) -> bool {
        matches!(self, Ok(_))
    }

    pub fn is_ok_and(self, f: impl FnOnce(T) -> bool) -> bool {
        self.ok().map(f).unwrap_or(false)
    }

    pub const fn is_any_err(&self) -> bool {
        !self.is_ok()
    }

    pub const fn is_recoverable(&self) -> bool {
        matches!(self, Recoverable(_))
    }

    pub fn is_recoverable_and(self, f: impl FnOnce(RE) -> bool) -> bool {
        self.recoverable().map(f).unwrap_or(false)
    }

    pub const fn is_fatal(&self) -> bool {
        matches!(self, Fatal(_))
    }

    pub fn is_fatal_and(self, f: impl FnOnce(FE) -> bool) -> bool {
        self.fatal().map(f).unwrap_or(false)
    }

    pub fn ok(self) -> Option<T> {
        match self {
            Ok(out) => Some(out),
            _ => None,
        }
    }

    pub fn recoverable(self) -> Option<RE> {
        match self {
            Recoverable(out) => Some(out),
            _ => None,
        }
    }

    pub fn fatal(self) -> Option<FE> {
        match self {
            Fatal(out) => Some(out),
            _ => None,
        }
    }

    pub fn non_fatal(self) -> Option<StdResult<T, RE>> {
        match self {
            Ok(out) => Some(StdOk(out)),
            Recoverable(err) => Some(StdErr(err)),
            Fatal(_) => None,
        }
    }

    pub fn err(self) -> Option<StdResult<RE, FE>> {
        match self {
            Ok(_) => None,
            Recoverable(err) => Some(StdOk(err)),
            Fatal(err) => Some(StdErr(err)),
        }
    }

    pub fn as_ref(&self) -> Result<&T, &RE, &FE> {
        match self {
            Ok(ok) => Ok(ok),
            Recoverable(err) => Recoverable(err),
            Fatal(err) => Fatal(err),
        }
    }

    pub fn as_mut(&mut self) -> Result<&mut T, &mut RE, &mut FE> {
        match self {
            Ok(ok) => Ok(ok),
            Recoverable(err) => Recoverable(err),
            Fatal(err) => Fatal(err),
        }
    }

    fn map_all<O, REO, FEO>(
        self,
        f: impl FnOnce(T) -> O,
        g: impl FnOnce(RE) -> REO,
        h: impl FnOnce(FE) -> FEO,
    ) -> Result<O, REO, FEO> {
        match self {
            Ok(ok) => Ok(f(ok)),
            Recoverable(err) => Recoverable(g(err)),
            Fatal(err) => Fatal(h(err)),
        }
    }

    pub fn map<O>(self, f: impl FnOnce(T) -> O) -> Result<O, RE, FE> {
        self.map_all(f, |x| x, |x| x)
    }

    pub fn map_or<O>(self, default: O, f: impl FnOnce(T) -> O) -> O {
        self.map(f).unwrap_or(default)
    }

    pub fn map_or_else<O>(self, default: impl FnOnce() -> O, f: impl FnOnce(T) -> O) -> O {
        self.map(f).unwrap_or_else(default)
    }

    pub fn map_recoverable<REO>(self, f: impl FnOnce(RE) -> REO) -> Result<T, REO, FE> {
        self.map_all(|x| x, f, |x| x)
    }

    pub fn map_fatal<FEO>(self, f: impl FnOnce(FE) -> FEO) -> Result<T, RE, FEO> {
        self.map_all(|x| x, |x| x, f)
    }

    pub fn inspect(self, f: impl FnOnce(&T)) -> Self {
        self.map(|x| {
            f(&x);
            x
        })
    }

    pub fn inspect_recoverable(self, f: impl FnOnce(&RE)) -> Self {
        self.map_recoverable(|x| {
            f(&x);
            x
        })
    }

    pub fn inspect_fatal(self, f: impl FnOnce(&FE)) -> Self {
        self.map_fatal(|x| {
            f(&x);
            x
        })
    }

    pub fn as_deref(&self) -> Result<&T::Target, &RE, &FE>
    where
        T: ops::Deref,
    {
        self.as_ref().map_deref()
    }

    pub fn as_deref_mut(&mut self) -> Result<&mut T::Target, &mut RE, &mut FE>
    where
        T: ops::DerefMut,
    {
        self.as_mut().map_deref_mut()
    }

    pub fn iter(&self) -> core::option::IntoIter<&T> {
        self.as_ref().ok().into_iter()
    }

    pub fn iter_mut(&mut self) -> core::option::IntoIter<&mut T> {
        self.as_mut().ok().into_iter()
    }

    pub fn expect_nonfatal(self, msg: &str) -> LocalResult<T, RE>
    where
        FE: fmt::Debug,
    {
        self.to_std().expect(msg).into()
    }

    pub fn expect_ok(self, msg: &str) -> T
    where
        RE: fmt::Debug,
        FE: fmt::Debug,
    {
        self.to_std().expect(msg).expect(msg)
    }

    pub fn unwrap_nonfatal(self) -> LocalResult<T, RE>
    where
        FE: fmt::Debug,
    {
        self.to_std().unwrap().into()
    }

    pub fn unwrap_ok(self) -> T
    where
        RE: fmt::Debug,
        FE: fmt::Debug,
    {
        self.to_std().unwrap().unwrap()
    }

    pub fn unwrap_or_default(self) -> T
    where
        T: Default,
    {
        self.unwrap_or_else(T::default)
    }

    pub fn unwrap_or(self, default: T) -> T {
        self.unwrap_or_else(move || default)
    }

    pub fn unwrap_or_else(self, default: impl FnOnce() -> T) -> T {
        match self {
            Ok(out) => out,
            _ => default(),
        }
    }

    pub fn and<O>(self, res: Result<O, RE, FE>) -> Result<O, RE, FE> {
        self.and_then(move |_| res)
    }

    pub fn and_then<O>(self, f: impl FnOnce(T) -> Result<O, RE, FE>) -> Result<O, RE, FE> {
        match self {
            Ok(out) => f(out),
            Recoverable(err) => Recoverable(err),
            Fatal(err) => Fatal(err),
        }
    }

    pub fn flatten_err<E: From<RE> + From<FE>>(self) -> StdResult<T, E> {
        match self {
            Ok(out) => StdOk(out),
            Recoverable(err) => StdErr(err.into()),
            Fatal(err) => StdErr(err.into()),
        }
    }

    pub fn convert_err<REO: From<RE>, FEO: From<FE>>(self) -> Result<T, REO, FEO> {
        match self {
            Ok(out) => Ok(out),
            Recoverable(err) => Recoverable(err.into()),
            Fatal(err) => Fatal(err.into()),
        }
    }

    pub fn more_fatal<REO>(self, f: impl FnOnce(RE) -> StdResult<REO, FE>) -> Result<T, REO, FE> {
        match self {
            Ok(out) => Ok(out),
            Recoverable(err) => match f(err) {
                StdOk(err) => Recoverable(err),
                StdErr(err) => Fatal(err),
            },
            Fatal(err) => Fatal(err),
        }
    }

    pub fn less_fatal<FEO>(self, f: impl FnOnce(FE) -> StdResult<RE, FEO>) -> Result<T, RE, FEO> {
        match self {
            Ok(out) => Ok(out),
            Recoverable(err) => Recoverable(err),
            Fatal(err) => match f(err) {
                StdOk(err) => Recoverable(err),
                StdErr(err) => Fatal(err),
            },
        }
    }

    pub fn collect_layered<U>(iter: impl IntoIterator<Item = Result<U, RE, FE>>) -> Self
    where
        T: FromIterator<U>,
    {
        let mut iter = iter.into_iter().map(Result::to_std);
        let mut out: StdResult<_, _> = match (&mut iter).collect() {
            StdOk(out) => out,
            StdErr(err) => return Fatal(err),
        };
        for i in iter {
            match i {
                StdOk(StdOk(_)) => (),
                StdOk(StdErr(recoverable)) if out.is_ok() => out = StdErr(recoverable),
                StdOk(StdErr(_)) => (),
                StdErr(err) => return Fatal(err),
            }
        }
        Self::ok_or_recoverable(out)
    }
}

impl<'a, T, RE, FE> Result<&'a T, RE, FE> {
    pub fn map_deref(self) -> Result<&'a <T as ops::Deref>::Target, RE, FE>
    where
        T: ops::Deref,
    {
        self.map(|x| &**x)
    }

    pub fn copied(self) -> Result<T, RE, FE>
    where
        T: Copy,
    {
        self.map(|ok| *ok)
    }

    pub fn cloned(self) -> Result<T, RE, FE>
    where
        T: Clone,
    {
        self.map(|ok| ok.clone())
    }
}

impl<'a, T, RE, FE> Result<&'a mut T, RE, FE> {
    pub fn map_deref(self) -> Result<&'a T::Target, RE, FE>
    where
        T: ops::DerefMut,
    {
        self.map(|x| &**x)
    }

    pub fn map_deref_mut(self) -> Result<&'a mut T::Target, RE, FE>
    where
        T: ops::DerefMut,
    {
        self.map(|x| &mut **x)
    }

    pub fn copied(self) -> Result<T, RE, FE>
    where
        T: Copy,
    {
        self.map(|ok| *ok)
    }

    pub fn cloned(self) -> Result<T, RE, FE>
    where
        T: Clone,
    {
        self.map(|ok| ok.clone())
    }
}

impl<T, RE, FE> Result<Result<T, RE, FE>, RE, FE> {
    pub fn flatten(self) -> Result<T, RE, FE> {
        self.and_then(|x| x)
    }
}

impl<T, RE, FE> From<Result<T, RE, FE>> for StdResult<StdResult<T, RE>, FE> {
    fn from(value: Result<T, RE, FE>) -> Self {
        value.to_std()
    }
}

impl<T, RE, FE> From<StdResult<StdResult<T, RE>, FE>> for Result<T, RE, FE> {
    fn from(value: StdResult<StdResult<T, RE>, FE>) -> Self {
        Self::from_std(value)
    }
}

impl<T, RE, FE> IntoIterator for Result<T, RE, FE> {
    type Item = T;
    type IntoIter = core::option::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.ok().into_iter()
    }
}

impl<'a, T, RE, FE> IntoIterator for &'a Result<T, RE, FE> {
    type Item = &'a T;
    type IntoIter = core::option::IntoIter<&'a T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, RE, FE> IntoIterator for &'a mut Result<T, RE, FE> {
    type Item = &'a mut T;
    type IntoIter = core::option::IntoIter<&'a mut T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, RE, FE, C> FromIterator<Result<T, RE, FE>> for Result<C, RE, FE>
where
    C: FromIterator<T>,
{
    fn from_iter<I: IntoIterator<Item = Result<T, RE, FE>>>(iter: I) -> Self {
        Self::from_std(iter.into_iter().map(Result::to_std).collect())
    }
}

impl<T, RE, FE, C, REC, FEC> FromIterator<Result<T, RE, FE>> for CollectedErrs<Result<C, REC, FEC>>
where
    C: FromIterator<T>,
    REC: FromIterator<RE>,
    FEC: FromIterator<FE>,
{
    fn from_iter<I: IntoIterator<Item = Result<T, RE, FE>>>(iter: I) -> Self {
        let collected: CollectedErrs<StdResult<CollectedErrs<StdResult<C, REC>>, FEC>> =
            iter.into_iter().map(Result::to_std).collect();

        CollectedErrs(collected.0.map(|collected| collected.0).into())
    }
}

impl<T, RE, FE, C, EC> FromIterator<Result<T, RE, FE>> for CollectedErrs<StdResult<C, EC>>
where
    C: FromIterator<T>,
    EC: FromIterator<StdResult<RE, FE>>,
{
    fn from_iter<I: IntoIterator<Item = Result<T, RE, FE>>>(iter: I) -> Self {
        iter.into_iter().map(Result::to_std_flipped).collect()
    }
}

impl<T, RE, FE, C, REC> FromIterator<Result<T, RE, FE>>
    for CollectedRecoverables<Result<C, REC, FE>>
where
    C: FromIterator<T>,
    REC: FromIterator<RE>,
{
    fn from_iter<I: IntoIterator<Item = Result<T, RE, FE>>>(iter: I) -> Self {
        let collected: StdResult<CollectedErrs<StdResult<C, REC>>, FE> =
            iter.into_iter().map(Result::to_std).collect();

        CollectedRecoverables(collected.map(|collected| collected.0).into())
    }
}

#[cfg(feature = "std")]
impl<T: std::process::Termination, RE: fmt::Debug, FE: fmt::Debug> std::process::Termination
    for Result<T, RE, FE>
{
    fn report(self) -> std::process::ExitCode {
        self.to_std().report()
    }
}

impl<T, RE, FE, FEO: From<FE>> ops::FromResidual<Result<Infallible, Infallible, FE>>
    for Result<T, RE, FEO>
{
    fn from_residual(residual: Result<Infallible, Infallible, FE>) -> Self {
        match residual {
            Ok(infallible) | Recoverable(infallible) => match infallible {},
            Fatal(err) => Fatal(err.into()),
        }
    }
}

impl<T, RE, FE> ops::Try for Result<T, RE, FE> {
    type Output = LocalResult<T, RE>;
    type Residual = Result<Infallible, Infallible, FE>;

    fn from_output(output: Self::Output) -> Self {
        match output {
            NoErr(ok) => Ok(ok),
            Handle(err) => Recoverable(err),
        }
    }

    fn branch(self) -> ops::ControlFlow<Self::Residual, Self::Output> {
        match self {
            Ok(ok) => ops::ControlFlow::Continue(NoErr(ok)),
            Recoverable(err) => ops::ControlFlow::Continue(Handle(err)),
            Fatal(err) => ops::ControlFlow::Break(Fatal(err)),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[must_use]
pub enum LocalResult<T, RE> {
    NoErr(T),
    Handle(RE),
}

pub use LocalResult::*;

impl<T, RE> LocalResult<T, RE> {
    pub fn from_std(res: StdResult<T, RE>) -> Self {
        match res {
            StdOk(ok) => NoErr(ok),
            StdErr(err) => Handle(err),
        }
    }

    pub fn to_std(self) -> StdResult<T, RE> {
        match self {
            NoErr(ok) => StdOk(ok),
            Handle(err) => StdErr(err),
        }
    }

    pub fn to_result<FE>(self) -> Result<T, RE, FE> {
        Result::ok_or_recoverable(self.to_std())
    }

    pub const fn is_ok(&self) -> bool {
        matches!(self, NoErr(_))
    }

    pub fn is_ok_and(self, f: impl FnOnce(T) -> bool) -> bool {
        self.ok().map(f).unwrap_or(false)
    }

    pub fn is_ok_or(self, f: impl FnOnce(RE) -> bool) -> bool {
        self.err().map(f).unwrap_or(true)
    }

    pub const fn is_err(&self) -> bool {
        !self.is_ok()
    }

    pub fn is_err_and(self, f: impl FnOnce(RE) -> bool) -> bool {
        self.err().map(f).unwrap_or(false)
    }

    pub fn is_err_or(self, f: impl FnOnce(T) -> bool) -> bool {
        self.ok().map(f).unwrap_or(true)
    }

    pub fn ok(self) -> Option<T> {
        match self {
            NoErr(out) => Some(out),
            Handle(_) => None,
        }
    }

    pub fn err(self) -> Option<RE> {
        match self {
            NoErr(_) => None,
            Handle(err) => Some(err),
        }
    }

    pub fn expect(self, msg: &str) -> T
    where
        RE: fmt::Debug,
    {
        self.to_std().expect(msg)
    }

    pub fn unwrap(self) -> T
    where
        RE: fmt::Debug,
    {
        self.to_std().unwrap()
    }
}

impl<T, RE> From<LocalResult<T, RE>> for StdResult<T, RE> {
    fn from(value: LocalResult<T, RE>) -> Self {
        value.to_std()
    }
}

impl<T, RE> From<StdResult<T, RE>> for LocalResult<T, RE> {
    fn from(value: StdResult<T, RE>) -> Self {
        Self::from_std(value)
    }
}

impl<T, RE, FE> From<LocalResult<T, RE>> for Result<T, RE, FE> {
    fn from(value: LocalResult<T, RE>) -> Self {
        value.to_result()
    }
}

/// only necessary because the `ops::Try` instance requires a `FromResidual` instance, this type is
/// *not* intended to be used as a return type.
impl<T, RE, REO: From<RE>> ops::FromResidual<LocalResult<Infallible, RE>> for LocalResult<T, REO> {
    fn from_residual(residual: LocalResult<Infallible, RE>) -> Self {
        match residual {
            NoErr(infallible) => match infallible {},
            Handle(err) => Handle(err.into()),
        }
    }
}

impl<T, RE> ops::Try for LocalResult<T, RE> {
    type Output = T;
    type Residual = LocalResult<Infallible, RE>;

    fn from_output(output: Self::Output) -> Self {
        NoErr(output)
    }

    fn branch(self) -> ops::ControlFlow<Self::Residual, Self::Output> {
        match self {
            NoErr(ok) => ops::ControlFlow::Continue(ok),
            Handle(err) => ops::ControlFlow::Break(Handle(err)),
        }
    }
}

// cross compatibility: -> StdResult<_, _>
impl<T, FE, FEO: From<FE>> ops::FromResidual<Result<Infallible, Infallible, FE>>
    for StdResult<T, FEO>
{
    fn from_residual(residual: Result<Infallible, Infallible, FE>) -> Self {
        match residual {
            Ok(infallible) | Recoverable(infallible) => match infallible {},
            Fatal(err) => StdResult::Err(err.into()),
        }
    }
}

impl<T, RE, REO: From<RE>, FEO> ops::FromResidual<LocalResult<Infallible, RE>>
    for StdResult<StdResult<T, REO>, FEO>
{
    fn from_residual(residual: LocalResult<Infallible, RE>) -> Self {
        match residual {
            NoErr(infallible) => match infallible {},
            Handle(err) => StdResult::Ok(StdResult::Err(err.into())),
        }
    }
}

// cross compatibility: -> Result<_, _, _>
impl<T, FE, REO, FEO: From<FE>> ops::FromResidual<StdResult<Infallible, FE>>
    for Result<T, REO, FEO>
{
    fn from_residual(residual: StdResult<Infallible, FE>) -> Self {
        match residual {
            StdOk(infallible) => match infallible {},
            StdErr(err) => Fatal(err.into()),
        }
    }
}

impl<T, RE, FE, REO: From<RE>> ops::FromResidual<LocalResult<Infallible, RE>>
    for Result<T, REO, FE>
{
    fn from_residual(residual: LocalResult<Infallible, RE>) -> Self {
        match residual {
            NoErr(infallible) => match infallible {},
            Handle(err) => Recoverable(err.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collected_errs_std() {
        type Collected = CollectedErrs<StdResult<Vec<u8>, Vec<&'static str>>>;

        let ok = StdOk::<u8, &'static str>;
        let err = StdErr::<u8, &'static str>;

        // empty
        assert_eq!(
            Vec::<StdResult<u8, &'static str>>::new()
                .into_iter()
                .collect::<Collected>(),
            CollectedErrs(StdOk(Vec::new())),
        );

        // only `Ok`
        assert_eq!(
            vec![ok(1), ok(2), ok(3)].into_iter().collect::<Collected>(),
            CollectedErrs(StdOk(vec![1, 2, 3]))
        );

        // only `Err`
        assert_eq!(
            vec![err("A"), err("B"), err("C")]
                .into_iter()
                .collect::<Collected>(),
            CollectedErrs(StdErr(vec!["A", "B", "C"])),
        );

        // interspliced `Err`
        assert_eq!(
            vec![ok(1), err("A"), err("B"), ok(2), err("C")]
                .into_iter()
                .collect::<Collected>(),
            CollectedErrs(StdErr(vec!["A", "B", "C"])),
        );
    }

    #[test]
    fn test_collected_layered() {
        type Collected = Result<Vec<u8>, &'static str, ()>;

        // empty
        assert_eq!(Collected::collect_layered([]), Ok(vec![]),);

        // only `Ok`
        assert_eq!(
            Collected::collect_layered([Ok(1), Ok(2), Ok(3)]),
            Ok(vec![1, 2, 3]),
        );

        // has `Recoverable`
        assert_eq!(
            Collected::collect_layered([Ok(1), Ok(2), Recoverable("X"), Ok(3)]),
            Recoverable("X"),
        );

        // has `Fatal` before `Recoverable`
        assert_eq!(
            Collected::collect_layered([Ok(1), Fatal(()), Ok(2), Recoverable("X"), Ok(3)]),
            Fatal(()),
        );

        // has `Fatal` after `Recoverable`, but still returns `Fatal`
        assert_eq!(
            Collected::collect_layered([Ok(1), Recoverable("X"), Ok(2), Fatal(()), Ok(3)]),
            Fatal(()),
        );
    }

    #[test]
    fn test_from_iter() {
        type Collected = Result<Vec<u8>, &'static str, ()>;

        // empty
        assert_eq!(Collected::from_iter([]), Ok(vec![]),);

        // only `Ok`
        assert_eq!(
            Collected::from_iter([Ok(1), Ok(2), Ok(3)]),
            Ok(vec![1, 2, 3]),
        );

        // has `Recoverable`
        assert_eq!(
            Collected::from_iter([Ok(1), Ok(2), Recoverable("X"), Ok(3)]),
            Recoverable("X"),
        );

        // has `Fatal` before `Recoverable`
        assert_eq!(
            Collected::from_iter([Ok(1), Fatal(()), Ok(2), Recoverable("X"), Ok(3)]),
            Fatal(()),
        );

        // has `Fatal` after `Recoverable`, but doesn't scan until that point
        assert_eq!(
            Collected::from_iter([Ok(1), Recoverable("X"), Ok(2), Fatal(()), Ok(3)]),
            Recoverable("X"),
        );
    }

    #[test]
    fn test_from_collected_errs() {
        type Collected = CollectedErrs<Result<Vec<u8>, Vec<&'static str>, Vec<()>>>;

        // empty
        assert_eq!(Collected::from_iter([]), CollectedErrs(Ok(vec![])),);

        // only `Ok`
        assert_eq!(
            Collected::from_iter([Ok(1), Ok(2), Ok(3)]),
            CollectedErrs(Ok(vec![1, 2, 3])),
        );

        // has `Recoverable`
        assert_eq!(
            Collected::from_iter([
                Ok(1),
                Ok(2),
                Recoverable("X"),
                Ok(3),
                Recoverable("Y"),
                Ok(4)
            ]),
            CollectedErrs(Recoverable(vec!["X", "Y"])),
        );

        // has `Fatal` before `Recoverable`
        assert_eq!(
            Collected::from_iter([Ok(1), Fatal(()), Ok(2), Recoverable("X"), Ok(3)]),
            CollectedErrs(Fatal(vec![()])),
        );

        // has `Fatal` after `Recoverable`
        assert_eq!(
            Collected::from_iter([Ok(1), Recoverable("X"), Ok(2), Fatal(()), Ok(3)]),
            CollectedErrs(Fatal(vec![()])),
        );
    }

    #[test]
    fn test_from_collected_errs_mixed() {
        type Collected = CollectedErrs<StdResult<Vec<u8>, Vec<StdResult<&'static str, ()>>>>;

        // empty
        assert_eq!(
            Collected::from_iter(Vec::<Result<u8, &'static str, ()>>::new()),
            CollectedErrs(StdOk(vec![])),
        );

        // only `Ok`
        assert_eq!(
            Collected::from_iter([Ok(1), Ok(2), Ok(3)]),
            CollectedErrs(StdOk(vec![1, 2, 3])),
        );

        // has `Recoverable`
        assert_eq!(
            Collected::from_iter([
                Ok(1),
                Ok(2),
                Recoverable("X"),
                Ok(3),
                Recoverable("Y"),
                Ok(4)
            ]),
            CollectedErrs(StdErr(vec![StdOk("X"), StdOk("Y")])),
        );

        // has `Fatal`
        assert_eq!(
            Collected::from_iter([Ok(1), Fatal(()), Ok(2), Recoverable("X"), Ok(3)]),
            CollectedErrs(StdErr(vec![StdErr(()), StdOk("X")])),
        );
    }

    #[test]
    fn test_from_collected_recoverables() {
        type Collected = CollectedRecoverables<Result<Vec<u8>, Vec<&'static str>, ()>>;

        // empty
        assert_eq!(Collected::from_iter([]), CollectedRecoverables(Ok(vec![])),);

        // only `Ok`
        assert_eq!(
            Collected::from_iter([Ok(1), Ok(2), Ok(3)]),
            CollectedRecoverables(Ok(vec![1, 2, 3])),
        );

        // has `Recoverable`
        assert_eq!(
            Collected::from_iter([
                Ok(1),
                Ok(2),
                Recoverable("X"),
                Ok(3),
                Recoverable("Y"),
                Ok(4)
            ]),
            CollectedRecoverables(Recoverable(vec!["X", "Y"])),
        );

        // has `Fatal` before `Recoverable`
        assert_eq!(
            Collected::from_iter([Ok(1), Fatal(()), Ok(2), Recoverable("X"), Ok(3)]),
            CollectedRecoverables(Fatal(())),
        );

        // has `Fatal` after `Recoverable`
        assert_eq!(
            Collected::from_iter([Ok(1), Recoverable("X"), Ok(2), Fatal(()), Ok(3)]),
            CollectedRecoverables(Fatal(())),
        );
    }

    mod type_checks {
        use super::*;

        struct OkTy;
        struct RecoverableTy;
        struct FatalTy;

        type NormalResultTy = StdResult<OkTy, FatalTy>;
        type StackedResultTy = StdResult<StdResult<OkTy, RecoverableTy>, FatalTy>;
        type ResultTy = Result<OkTy, RecoverableTy, FatalTy>;

        fn std_result() -> NormalResultTy {
            unimplemented!()
        }
        fn result() -> ResultTy {
            unimplemented!()
        }

        mod tests {
            #![allow(dead_code)]
            use super::*;

            fn returns_std_result() -> StackedResultTy {
                std_result()?;
                match result()? {
                    NoErr(_) | Handle(_) => (),
                }
                result()??;

                StdOk(StdOk(OkTy))
            }

            fn returns_result() -> ResultTy {
                std_result()?;
                match result()? {
                    NoErr(_) | Handle(_) => (),
                }
                result()??;

                Ok(OkTy)
            }

            /// This should warn because `result()?` still has an unhandled error
            fn this_should_warn() -> ResultTy { result()?; Ok(OkTy) }
        }
    }
}
