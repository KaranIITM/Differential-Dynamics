# -*- coding: utf-8 -*-
"""DE Solver.ipynb

Original file is located at
    https://colab.research.google.com/drive/1dMZO5mOO4L-EN4d-0evE1ZJKX0DlBY-k
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

st.set_page_config(page_title="Differential-Equation Solver",
                   layout="wide", page_icon="")

##############################################################################
# ---------------------------  FRONT  PAGE  -------------------------------- #
##############################################################################
home, ode_tab, pde_tab, feedback = st.tabs(
    [" Home", "ODE Solver", "PDE Solver", "Feedback"])

##############################################################################
# -----------------------  FRONT-PAGE CONTENT  ----------------------------- #
##############################################################################
with home:
    st.title("Differential-Equation Solver: ODE and PDE Methods")
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        if st.button("ODE Solver"):
            st.query_params.tab = "ODE"  # FIXED: Updated to new API
    with col2:
        if st.button("PDE Solver"):
            st.query_params.tab = "PDE"  # FIXED: Updated to new API
    with col3:
        if st.button("Feedback"):
            st.query_params.tab = "FB"   # FIXED: Updated to new API

    st.subheader("Where these solvers change the world")
    sectors = {
        "Aerospace Engineering":
            "Trajectory optimisation, flight-dynamics simulation",
        "Automotive Industry":
            "CFD for aerodynamics and battery thermal management",
        "Biomedical Engineering":
            "Drug-release modelling, tissue-growth PDEs",
        "Climate Science":
            "Global circulation & weather forecasting PDEs",
        "Financial Engineering":
            "Option-pricing (Black–Scholes) and risk ODE/PDEs",
        "Energy Systems":
            "Reactor dynamics, wind-farm optimisation"}
    st.table(pd.DataFrame(list(sectors.items()),
                          columns=["Sector", "Why Differential-Eqn Tools Matter"]))

##############################################################################
# -----------------------------  ODE  TAB  --------------------------------- #
##############################################################################
with ode_tab:
    st.header("Solve Ordinary Differential Equations")
    ode_col1, ode_col2 = st.columns([1, 2])

    with ode_col1:
        method = st.selectbox("Numerical Method", [
            "Euler", "RK2", "RK4", "RKF45 (adaptive)",
            "Adams-Bashforth 4", "Adams-Moulton 4",
            "BDF2", "Verlet (2nd-order)", "Stormer-Verlet (symplectic)"])
        ode_str = st.text_area("Enter system  f(t, X)  as python lambda",
                               value="lambda t, X: [X[1], -X]")  # FIXED: Added 
        y0_str = st.text_input("Initial conditions (comma-separated)", "1, 0")
        t0 = st.number_input("t₀", value=0.0)
        t1 = st.number_input("t final", value=10.0, min_value=t0+1e-6)
        h = st.number_input("Step size (initial if adaptive)", value=0.1,
                            min_value=1e-6, max_value=5.0)
        compute = st.button("Run ODE Solver")

    def parse_ic(text):
        try:
            return np.array([float(x) for x in text.split(",")])
        except Exception:
            st.error("Invalid initial conditions")
            st.stop()

    # -----------------  NUMERICAL  METHODS  ---------------- #
    def euler(f, t0, y0, h, n):
        T = np.linspace(t0, t0+n*h, n+1)
        Y = np.zeros((n+1, len(y0)))
        Y[0] = y0  # FIXED: Proper initialization
        for i in range(n):
            Y[i+1] = Y[i] + h * np.array(f(T[i], Y[i]))
        return T, Y

    def rk2(f, t0, y0, h, n):
        T = np.linspace(t0, t0+n*h, n+1)
        Y = np.zeros((n+1, len(y0)))
        Y = y0  # FIXED: Proper initialization
        for i in range(n):
            k1 = np.array(f(T[i], Y[i]))
            k2 = np.array(f(T[i]+h/2, Y[i]+h*k1/2))
            Y[i+1] = Y[i] + h * k2
        return T, Y

    def rk4(f, t0, y0, h, n):
        T = np.linspace(t0, t0+n*h, n+1)
        Y = np.zeros((n+1, len(y0)))
        Y = y0  # FIXED: Proper initialization
        for i in range(n):
            k1 = np.array(f(T[i], Y[i]))
            k2 = np.array(f(T[i]+h/2, Y[i]+h*k1/2))
            k3 = np.array(f(T[i]+h/2, Y[i]+h*k2/2))
            k4 = np.array(f(T[i]+h,   Y[i]+h*k3))
            Y[i+1] = Y[i] + h*(k1+2*k2+2*k3+k4)/6
        return T, Y

    def rkf45(f, t0, y0, t_end, h, tol=1e-6):
        T, Y, t = [t0], [y0], t0
        y = y0
        while t < t_end:
            if t+h > t_end:
                h = t_end - t
            k1 = np.array(f(t, y))
            k2 = np.array(f(t + h/4,          y + h*k1/4))
            k3 = np.array(f(t + 3*h/8,        y + h*(3*k1+9*k2)/32))
            k4 = np.array(f(t + 12*h/13,      y + h*(1932*k1 -7200*k2 +7296*k3)/2197))
            k5 = np.array(f(t + h,            y + h*(439*k1/216 -8*k2 +3680*k3/513 -845*k4/4104)))
            k6 = np.array(f(t + h/2,          y + h*(-8*k1/27 +2*k2 -3544*k3/2565
                                             +1859*k4/4104 -11*k5/40)))
            y4 = y + h*(25*k1/216 +1408*k3/2565 +2197*k4/4104 -k5/5)
            y5 = y + h*(16*k1/135 +6656*k3/12825 +28561*k4/56430
                        -9*k5/50 +2*k6/55)
            error = np.linalg.norm(y5 - y4)
            if error < tol or h < 1e-12:
                t += h
                y = y5
                T.append(t)
                Y.append(y)
            h *= min(2, max(0.5, 0.9*(tol/error)**0.2))
        return np.array(T), np.array(Y)

    # start-up with RK4 for multistep schemes
    def start_rk4(f, t0, y0, h):
        return rk4(f, t0, y0, h, 3)

    def adams_bashforth4(f, t0, y0, h, n):
        T, Y = start_rk4(f, t0, y0, h)
        for i in range(3, n):
            t = T[-1] + h
            y = Y[-1] + h/24*(55*np.array(f(T[-1], Y[-1])) -59*np.array(f(T[-2], Y[-2]))
                              +37*np.array(f(T[-3], Y[-3])) -9*np.array(f(T[-4], Y[-4])))
            T = np.append(T, t)
            Y = np.vstack([Y, y])
        return T, Y

    def adams_moulton4(f, t0, y0, h, n):
        T, Y = start_rk4(f, t0, y0, h)
        for i in range(3, n):
            t_pred = T[-1] + h
            y_pred = Y[-1] + h/24*(55*np.array(f(T[-1], Y[-1])) -59*np.array(f(T[-2], Y[-2]))
                                   +37*np.array(f(T[-3], Y[-3])) -9*np.array(f(T[-4], Y[-4])))
            y_corr = Y[-1] + h/24*(9*np.array(f(t_pred, y_pred)) +19*np.array(f(T[-1], Y[-1]))
                                   -5*np.array(f(T[-2], Y[-2])) +np.array(f(T[-3], Y[-3])))  # FIXED: Added missing np.array()
            T = np.append(T, t_pred)
            Y = np.vstack([Y, y_corr])
        return T, Y

    def bdf2(f, t0, y0, h, n, its=5):
        T = [t0, t0+h]
        Y = [y0, y0 + h*np.array(f(t0, y0))]
        for i in range(1, n):
            t_next = T[-1] + h
            y = Y[-1]           # initial guess
            for _ in range(its):
                y = (4*Y[-1] -Y[-2] +2*h*np.array(f(t_next, y)))/3
            T.append(t_next)
            Y.append(y)
        return np.array(T), np.array(Y)

    def verlet(accel, t0, y0, h, n):
        if len(y0) != 2:  # FIXED: Added validation
            st.error("Verlet method requires exactly 2 initial conditions [position, velocity]")
            st.stop()
        T = np.linspace(t0, t0+n*h, n+1)
        Y = np.zeros((n+1, 2))
        Y[0] = y0  # FIXED: Proper initialization
        a0 = accel(t0, Y[0, 0])
        Y[1, 0] = Y[0, 0] + Y[0, 1]*h + 0.5*a0*h**2
        Y[1, 1] = Y[0, 1] + 0.5*(a0 + accel(t0+h, Y[1, 0]))*h
        for i in range(1, n):
            a = accel(T[i], Y[i, 0])
            Y[i+1, 0] = 2*Y[i, 0] - Y[i-1, 0] + a*h**2
            a_next = accel(T[i+1], Y[i+1, 0])
            Y[i+1, 1] = Y[i, 1] + 0.5*(a + a_next)*h
        return T, Y

    def stormer_verlet(force, t0, y0, h, n):
        if len(y0) != 2:  # FIXED: Added validation
            st.error("Stormer-Verlet method requires exactly 2 initial conditions [position, momentum]")
            st.stop()
        T = np.linspace(t0, t0+n*h, n+1)
        Y = np.zeros((n+1, 2))
        Y[0] = y0  # FIXED: Proper initialization
        p_half = Y[0, 1] + 0.5*h*force(t0, Y[0, 0])
        for i in range(n):
            q_next = Y[i, 0] + h*p_half
            p_next = p_half + 0.5*h*force(T[i+1], q_next)
            Y[i+1] = [q_next, p_next]
            p_half = p_next + 0.5*h*force(T[i+1], q_next)
        return T, Y

    # ------------------------  RUN  BUTTON  ---------------- #
    if compute:
        y0 = parse_ic(y0_str)
        n_steps = int(np.ceil((t1-t0)/h))
        
        # FIXED: Added missing function evaluation
        try:
            f = eval(ode_str, {"np": np})
        except Exception as err:
            st.error(f"Invalid function: {err}")
            st.stop()
        
        # FIXED: Added proper validation
        try:
            test_output = f(t0, y0)
            if len(test_output) != len(y0):
                st.error(f"Function returns {len(test_output)} values but initial conditions have {len(y0)} values")
                st.stop()
        except Exception as e:
            st.error(f"Function evaluation failed: {e}")
            st.stop()

        if method == "Euler":
            T, Y = euler(f, t0, y0, h, n_steps)
        elif method == "RK2":
            T, Y = rk2(f, t0, y0, h, n_steps)
        elif method == "RK4":
            T, Y = rk4(f, t0, y0, h, n_steps)
        elif method == "RKF45 (adaptive)":
            T, Y = rkf45(f, t0, y0, t1, h)
        elif method == "Adams-Bashforth 4":
            T, Y = adams_bashforth4(f, t0, y0, h, n_steps)
        elif method == "Adams-Moulton 4":
            T, Y = adams_moulton4(f, t0, y0, h, n_steps)
        elif method == "BDF2":
            T, Y = bdf2(f, t0, y0, h, n_steps)
        elif method == "Verlet (2nd-order)":
            accel = lambda t, x: f(t, np.array([x, 0]))[1]
            T, Y = verlet(accel, t0, y0, h, n_steps)
        else:   # Stormer-Verlet
            force = lambda t, q: f(t, np.array([q, 0]))[1]
            T, Y = stormer_verlet(force, t0, y0, h, n_steps)

        st.success("Computation complete")
        # ------------------  PLOTS  ----------------------- #
        ode_col2.subheader("Solution curves")
        fig, ax = plt.subplots()
        for i in range(Y.shape[1]):
            ax.plot(T, Y[:, i], label=f"x{i}")
        ax.set_xlabel("t"); ax.grid(); ax.legend()
        ode_col2.pyplot(fig, use_container_width=True)

        if Y.shape[1] >= 2:
            ode_col2.subheader("Phase portrait (x0 vs x1)")
            fig2, ax2 = plt.subplots()
            ax2.plot(Y[:, 0], Y[:, 1])
            ax2.set_xlabel("x0"); ax2.set_ylabel("x1"); ax2.grid()
            ode_col2.pyplot(fig2, use_container_width=True)

        ode_col2.subheader("Numerical Data")
        df = pd.DataFrame(Y, columns=[f"x{i}" for i in range(Y.shape[1])])
        df.insert(0, "t", T)
        ode_col2.dataframe(df)

##############################################################################
# -----------------------------  PDE  TAB  --------------------------------- #
##############################################################################
with pde_tab:
    st.header("Solve Classic 1-D PDEs (Finite Difference)")

    pde_type = st.selectbox("PDE", ["Heat Equation", "Wave Equation", "Laplace Equation"])
    scheme = st.selectbox("Scheme", ["Forward-Euler", "Backward-Euler",
                                     "Crank-Nicolson", "FTCS (wave)", "Method of Lines"])
    L = st.number_input("Domain length L", value=1.0)
    Nx = st.number_input("Spatial nodes Nx", value=51, step=1, min_value=3)
    Nt = st.number_input("Time steps Nt", value=200, step=1, min_value=1)
    dt = st.number_input("Δt (time step)", value=0.001)
    alpha = st.number_input("Diffusion/C² (heat α, wave c²)", value=1.0)
    ic = st.text_input("Initial condition  u0(x)", "np.sin(np.pi*x/L)")
    bc_left = st.text_input("Boundary u(0,t)", "0")
    bc_right = st.text_input("Boundary u(L,t)", "0")
    run_pde = st.button("Run PDE Solver")

    if run_pde:
        x = np.linspace(0, L, int(Nx))
        dx = x[1]-x  # FIXED: Added missing 
        u = eval(ic, {"np": np, "x": x, "L": L})
        u_all = [u.copy()]

        if pde_type == "Heat Equation" and scheme in {"Forward-Euler", "Crank-Nicolson", "Backward-Euler"}:
            lam = alpha*dt/dx**2
            if lam > 0.5 and scheme == "Forward-Euler":
                st.warning("Scheme unstable (α Δt/Δx² > 0.5)")
            A = np.diag((1-2*lam)*np.ones(len(x))) + np.diag(lam*np.ones(len(x)-1), 1) + np.diag(lam*np.ones(len(x)-1), -1)

            if scheme == "Backward-Euler":
                A = np.diag((1+2*lam)*np.ones(len(x))) - np.diag(lam*np.ones(len(x)-1), 1) - np.diag(lam*np.ones(len(x)-1), -1)
                A_inv = np.linalg.inv(A)
            if scheme == "Crank-Nicolson":
                B = np.diag((1-2*lam)*np.ones(len(x))) + np.diag(lam*np.ones(len(x)-1), 1) + np.diag(lam*np.ones(len(x)-1), -1)
                C = np.diag((1+2*lam)*np.ones(len(x))) - np.diag(lam*np.ones(len(x)-1), 1) - np.diag(lam*np.ones(len(x)-1), -1)
                C_inv = np.linalg.inv(C)

            for _ in range(int(Nt)):
                # FIXED: Proper boundary condition handling
                try:
                    u[0] = float(eval(bc_left, {"t": _*dt, "np": np}))
                    u[-1] = float(eval(bc_right, {"t": _*dt, "np": np}))
                except:
                    u = 0  # default boundary
                    u[-1] = 0

                if scheme == "Forward-Euler":
                    u = A @ u
                elif scheme == "Backward-Euler":
                    u = A_inv @ u
                else:   # Crank-Nicolson
                    u = C_inv @ (B @ u)
                u_all.append(u.copy())

        elif pde_type == "Wave Equation" and scheme == "FTCS (wave)":
            lam = alpha*dt**2/dx**2
            if lam > 1:
                st.warning("Scheme unstable (c² Δt²/Δx² > 1)")
            u_prev = u.copy()
            u = u.copy()
            for _ in range(int(Nt)):
                u_next = np.zeros_like(u)
                u_next[1:-1] = 2*u[1:-1] - u_prev[1:-1] + lam*(u[2:] -2*u[1:-1] + u[:-2])
                u_prev = u.copy()
                u = u_next.copy()
                u_all.append(u.copy())

        elif pde_type == "Laplace Equation":
            # steady-state solution via Gauss-Seidel
            u = np.zeros_like(x)
            for _ in range(int(Nt)):
                u[0] = float(eval(bc_left, {"np": np}))  # FIXED: Added proper evaluation
                u[-1] = float(eval(bc_right, {"np": np}))
                for i in range(1, len(x)-1):
                    u[i] = 0.5*(u[i-1] + u[i+1])
            u_all = [u]

        elif scheme == "Method of Lines":
            # semi-discrete heat eqn  -> ODE system
            lam = alpha/dx**2
            A = np.diag(-2*np.ones(len(x))) + np.diag(np.ones(len(x)-1), 1) + np.diag(np.ones(len(x)-1), -1)
            def rhs(_, y): return lam*A @ y
            T, Y = rk4(rhs, 0, u, dt, int(Nt))
            u_all = Y

        # ---------------  PLOTS  --------------- #
        st.success("PDE solved")
        fig = plt.figure()
        xx, tt = np.meshgrid(x, np.arange(len(u_all))*dt)
        plt.contourf(xx, tt, np.array(u_all), 50, cmap="viridis")
        plt.colorbar(label="u")
        plt.xlabel("x"); plt.ylabel("t")
        st.pyplot(fig, use_container_width=True)

##############################################################################
# ---------------------------  FEEDBACK  TAB  ------------------------------ #
##############################################################################
with feedback:
    st.header("Send us feedback")
    st.markdown(
        """
        <form action="https://formsubmit.co/ma24m012@smail.iitm.ac.in" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required> <br><br>
            <input type="email" name="email" placeholder="Your email" required> <br><br>
            <input type="text" name="_subject" placeholder="Subject" required> <br><br>
            <textarea name="message" rows="6" placeholder="Your message" required></textarea><br><br>
            <button type="submit">Send</button>
        </form>
        """, unsafe_allow_html=True)
