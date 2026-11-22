import { FormEvent, ReactNode, useEffect, useState } from "react";
import {
  createUserWithEmailAndPassword,
  onAuthStateChanged,
  sendEmailVerification,
  signInWithEmailAndPassword,
  signOut,
  updateProfile,
  type User,
} from "firebase/auth";
import kpmgBlueLogo from "./assets/kpmg-logo-blue.svg";
import { fbAuth } from "./firebase";



function friendlyError(error: unknown): string {
  const code = (error as { code?: string })?.code;
  switch (code) {
    case "auth/email-already-in-use":
      return "An account with this email already exists. Try logging in.";
    case "auth/invalid-credential":
    case "auth/wrong-password":
    case "auth/user-not-found":
      return "Incorrect email or password.";
    case "auth/weak-password":
      return "Choose a stronger password (at least 8 characters).";
    case "auth/too-many-requests":
      return "Too many attempts. Wait a few minutes and try again.";
    case "auth/network-request-failed":
      return "Network error. Check your connection.";
    default:
      return error instanceof Error ? error.message : "Something went wrong.";
  }
}

function AuthShell({ children }: { children: ReactNode }) {
  return (
    <main className="auth-shell">
      <div className="auth-card">
        <img className="auth-logo" src={kpmgBlueLogo} alt="KPMG" />
        {children}
      </div>
    </main>
  );
}

export default function AuthGate({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [verified, setVerified] = useState(false);
  const [ready, setReady] = useState(false);
  const [mode, setMode] = useState<"signup" | "login">("signup");
  const [form, setForm] = useState({ first: "", last: "", email: "", password: "" });
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(
    () =>
      onAuthStateChanged(fbAuth, (u) => {
        setUser(u);
        setVerified(!!u?.emailVerified);
        setReady(true);
      }),
    [],
  );

  const run = async (fn: () => Promise<void>) => {
    setMsg("");
    setBusy(true);
    try {
      await fn();
    } catch (e) {
      setMsg(friendlyError(e));
    } finally {
      setBusy(false);
    }
  };

  const submit = (e: FormEvent) => {
    e.preventDefault();
    run(async () => {
      const email = form.email.trim();
      if (mode === "signup") {
        const cred = await createUserWithEmailAndPassword(fbAuth, email, form.password);
        await updateProfile(cred.user, {
          displayName: `${form.first.trim()} ${form.last.trim()}`,
        });
        await sendEmailVerification(cred.user);
        setMsg("Verification link sent. Check your inbox and spam folder.");
      } else {
        await signInWithEmailAndPassword(fbAuth, email, form.password);
      }
    });
  };

  const checkVerified = () =>
    run(async () => {
      const current = fbAuth.currentUser;
      if (!current) return;
      await current.reload();
      await current.getIdToken(true); // refresh so the API sees email_verified = true
      const ok = !!current.emailVerified;
      setVerified(ok);
      if (!ok) setMsg("Not verified yet. Click the link in the email, then try again.");
    });

  const resend = () =>
    run(async () => {
      if (!fbAuth.currentUser) return;
      await sendEmailVerification(fbAuth.currentUser);
      setMsg("Link resent. Check your inbox and spam folder.");
    });

  if (!ready) return null;
  if (user && verified) return <>{children}</>;

  if (user) {
    return (
      <AuthShell>
        <h2>Verify your email</h2>
        <p>
          We sent a verification link to <strong>{user.email}</strong>. Click it, then come back
          here.
        </p>
        <button disabled={busy} onClick={checkVerified}>
          I've verified, continue
        </button>
        <button type="button" className="auth-secondary" disabled={busy} onClick={resend}>
          Resend link
        </button>
        <button type="button" className="auth-secondary" onClick={() => signOut(fbAuth)}>
          Use a different account
        </button>
        {msg && <p className="auth-message">{msg}</p>}
      </AuthShell>
    );
  }

  return (
    <AuthShell>
      <form onSubmit={submit}>
        <h2>{mode === "signup" ? "Create your account" : "Log in"}</h2>
        {mode === "signup" && (
          <>
            <input
              placeholder="First name"
              autoComplete="given-name"
              value={form.first}
              onChange={(e) => setForm({ ...form, first: e.target.value })}
              required
            />
            <input
              placeholder="Last name"
              autoComplete="family-name"
              value={form.last}
              onChange={(e) => setForm({ ...form, last: e.target.value })}
              required
            />
          </>
        )}
        <input
          type="email"
          placeholder="Email address"
          autoComplete="email"
          value={form.email}
          onChange={(e) => setForm({ ...form, email: e.target.value })}
          required
        />
        <input
          type="password"
          placeholder="Password"
          autoComplete={mode === "signup" ? "new-password" : "current-password"}
          minLength={8}
          value={form.password}
          onChange={(e) => setForm({ ...form, password: e.target.value })}
          required
        />
        <button disabled={busy}>{mode === "signup" ? "Sign up" : "Log in"}</button>
        <button
          type="button"
          className="auth-secondary"
          onClick={() => {
            setMode(mode === "signup" ? "login" : "signup");
            setMsg("");
          }}
        >
          {mode === "signup" ? "Have an account? Log in" : "New here? Sign up"}
        </button>
        {msg && <p className="auth-message">{msg}</p>}
      </form>
    </AuthShell>
  );
}