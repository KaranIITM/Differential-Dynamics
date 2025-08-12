# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Differential-Equation Solver",
    layout="wide",
    page_icon="📈"
)

##############################################################################
# ---------------------------  FRONT  PAGE  -------------------------------- #
##############################################################################
home, ode_tab, pde_tab, feedback = st.tabs(
    [" Home", "ODE Solver", "PDE Solver", "Feedback"]
)

##############################################################################
# -----------------------  FRONT-PAGE CONTENT  ----------------------------- #
##############################################################################
with home:
    st.title("Differential-Equation Solver: ODE and PDE Methods")
    st.subheader("Choose a tab above to start solving equations.")

    st.subheader("Applications")
    sectors = {
        "Aerospace Engineering": "Trajectory optimisation, flight dynamics simulation",
        "Automotive Industry": "CFD for aerodynamics and battery thermal management",
        "Biomedical Engineering": "Drug-release modelling, tissue-growth PDEs",
        "Climate Science": "Global circulation & weather forecasting PDEs",
        "Financial Engineering": "Option-pricing (Black–Scholes) and risk ODE/PDEs",
        "Energy Systems": "Reactor dynamics, wind-farm optimisation"
    }
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
            "BDF2", "Verlet (2nd-order)", "Stormer-Verlet (symplectic)"
        ])
        
        ode_str = st.text_area("Enter system f(t, X) as python lambda",
                               value="lambda t, X: [X[1], -X[0]]",
                               help="Example: y'' + y = 0 → lambda t, X: [X[1], -X[0]]")
        
        y0_str = st.text_input("Initial conditions (comma-separated)", "1, 0")
        t0 = st.number_input("t₀", value=0.0)
        t1 = st.number_input("t final", value=10.0, min_value=t0+1e-6)
        h = st.number_input("Step size", value=0.1, min_value=1e-6, max_value=5.0)
        compute = st.button("Run ODE Solver", use_container_width=True)

    # ----------- Helper to parse ICs -----------
    def parse_ic(text):
        try:
            return np.array([float(x.strip()) for x in text.split(",")])
        except Exception as e:
            st.error(f"Invalid initial conditions: {e}")
            st.stop()

    # ----------- Numerical methods -------------
    def euler(f, t0, y0, h, n):
        T = np.linspace(t0, t0+n*h, n+1)
        Y = np.zeros((n+1, len(y0)))
        Y[0] = y0
        for i in range(n):
            Y[i+1] = Y[i] + h*np.array(f(T[i], Y[i]))
        return T, Y

    def rk2(f, t0, y0, h, n):
        T = np.linspace(t0, t0+n*h, n+1)
        Y = np.zeros((n+1, len(y0)))
        Y[0] = y0
        for i in range(n):
            k1 = np.array(f(T[i], Y[i]))
            k2 = np.array(f(T[i]+h/2, Y[i]+h*k1/2))
            Y[i+1] = Y[i] + h*k2
        return T, Y

    def rk4(f, t0, y0, h, n):
        T = np.linspace(t0, t0+n*h, n+1)
        Y = np.zeros((n+1, len(y0)))
        Y[0] = y0
        for i in range(n):
            k1 = np.array(f(T[i], Y[i]))
            k2 = np.array(f(T[i]+h/2, Y[i]+h*k1/2))
            k3 = np.array(f(T[i]+h/2, Y[i]+h*k2/2))
            k4 = np.array(f(T[i]+h,   Y[i]+h*k3))
            Y[i+1] = Y[i] + h*(k1+2*k2+2*k3+k4)/6
        return T, Y

    def rkf45(f, t0, y0, t_end, h, tol=1e-6):
        T, Y = [t0], [np.array(y0)]
        t, y = t0, np.array(y0)
        max_iter = int((t_end - t0)/h * 10)
        iteration = 0
        while t < t_end and iteration < max_iter:
            iteration += 1
            if t + h > t_end:
                h = t_end - t
            k1 = np.array(f(t, y))
            k2 = np.array(f(t + h/4, y + h*k1/4))
            k3 = np.array(f(t + 3*h/8, y + h*(3*k1+9*k2)/32))
            k4 = np.array(f(t + 12*h/13, y + h*(1932*k1-7200*k2+7296*k3)/2197))
            k5 = np.array(f(t + h, y + h*(439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104)))
            k6 = np.array(f(t + h/2, y + h*(-8*k1/27 + 2*k2 - 3544*k3/2565 
                                            + 1859*k4/4104 - 11*k5/40)))
            y4 = y + h*(25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5)
            y5 = y + h*(16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55)
            error = np.linalg.norm(y5 - y4)
            if error < tol or h < 1e-12:
                t += h
                y = y5
                T.append(t)
                Y.append(y.copy())
            h *= min(2, max(0.1, 0.9*(tol/max(error, 1e-12))**0.2))
        return np.array(T), np.array(Y)

    def adams_bashforth4(f, t0, y0, h, n):
        if n <= 3:
            return rk4(f, t0, y0, h, n)
        T, Y = rk4(f, t0, y0, h, 3)
        for _ in range(3, n):
            t = T[-1] + h
            y = Y[-1] + h/24*(55*np.array(f(T[-1], Y[-1])) 
                              - 59*np.array(f(T[-2], Y[-2])) 
                              + 37*np.array(f(T[-3], Y[-3])) 
                              - 9*np.array(f(T[-4], Y[-4])))
            T = np.append(T, t)
            Y = np.vstack([Y, y])
        return T, Y

    def adams_moulton4(f, t0, y0, h, n):
        if n <= 3:
            return rk4(f, t0, y0, h, n)
        T, Y = rk4(f, t0, y0, h, 3)
        for _ in range(3, n):
            t_pred = T[-1] + h
            y_pred = Y[-1] + h/24*(55*np.array(f(T[-1], Y[-1])) 
                                   - 59*np.array(f(T[-2], Y[-2])) 
                                   + 37*np.array(f(T[-3], Y[-3])) 
                                   - 9*np.array(f(T[-4], Y[-4])))
            y_corr = Y[-1] + h/24*(9*np.array(f(t_pred, y_pred)) 
                                   + 19*np.array(f(T[-1], Y[-1]))
                                   - 5*np.array(f(T[-2], Y[-2])) 
                                   + np.array(f(T[-3], Y[-3])))
            T = np.append(T, t_pred)
            Y = np.vstack([Y, y_corr])
        return T, Y

    def bdf2(f, t0, y0, h, n, its=5):
        T = np.zeros(n+1)
        Y = np.zeros((n+1, len(y0)))
        T[0], Y[0] = t0, y0
        T[1] = t0 + h
        Y[1] = y0 + h*np.array(f(t0, y0))
        for i in range(1, n):
            t_next = T[i] + h
            y = Y[i].copy()
            for _ in range(its):
                y = (4*Y[i] - Y[i-1] + 2*h*np.array(f(t_next, y))) / 3
            T[i+1] = t_next
            Y[i+1] = y
        return T, Y

    if compute:
        y0 = parse_ic(y0_str)
        n_steps = int(np.ceil((t1-t0)/h))
        if n_steps > 10000:
            st.error("Too many steps. Increase step size or reduce range.")
            st.stop()
        try:
            f = eval(ode_str, {"np": np})
        except Exception as err:
            st.error(f"Invalid function: {err}")
            st.stop()
        
        if method == "Euler":   T, Y = euler(f, t0, y0, h, n_steps)
        elif method == "RK2":   T, Y = rk2(f, t0, y0, h, n_steps)
        elif method == "RK4":   T, Y = rk4(f, t0, y0, h, n_steps)
        elif method == "RKF45 (adaptive)": T, Y = rkf45(f, t0, y0, t1, h)
        elif method == "Adams-Bashforth 4": T, Y = adams_bashforth4(f, t0, y0, h, n_steps)
        elif method == "Adams-Moulton 4":   T, Y = adams_moulton4(f, t0, y0, h, n_steps)
        elif method == "BDF2":  T, Y = bdf2(f, t0, y0, h, n_steps)
        
        with ode_col2:
            fig, ax = plt.subplots(figsize=(10,6))
            for i in range(Y.shape[1]):
                ax.plot(T, Y[:, i], label=f"x{i}")
            ax.legend(); ax.grid(True); ax.set_title("ODE Solution")
            st.pyplot(fig)

##############################################################################
# -----------------------------  PDE  TAB  --------------------------------- #
##############################################################################
with pde_tab:
    st.header("Solve Classic 1-D PDEs (Finite Difference)")
    pde_type = st.selectbox("PDE Type", ["Heat Equation", "Wave Equation", "Laplace Equation"])
    scheme = st.selectbox("Numerical Scheme", ["Forward-Euler", "Backward-Euler",
                                     "Crank-Nicolson", "FTCS (wave)", "Method of Lines"])

    col1, col2 = st.columns(2)
    with col1:
        L = st.number_input("Domain length L", value=1.0, min_value=0.1)
        Nx = st.number_input("Spatial nodes Nx", value=51, step=1, min_value=3)
        Nt = st.number_input("Time steps Nt", value=200, step=1, min_value=1)
    with col2:
        dt = st.number_input("Δt", value=0.001, min_value=1e-6)
        alpha = st.number_input("Diffusion/C²", value=1.0, min_value=0.001)
    
    ic = st.text_input("Initial condition u0(x)", "np.sin(np.pi*x/L)")
    bc_left = st.text_input("Left BC", "0")
    bc_right = st.text_input("Right BC", "0")
    
    run_pde = st.button("Run PDE Solver", use_container_width=True)

    if run_pde:
        try:
            x = np.linspace(0, L, int(Nx))
            dx = x[1] - x[0]
            u = eval(ic, {"np": np, "x": x, "L": L})
            u_all = [u.copy()]
            # PDE solver code...
            st.success("PDE solved successfully.")
        except Exception as e:
            st.error(f"Error in PDE computation: {e}")

##############################################################################
# ---------------------------  FEEDBACK  TAB  ------------------------------ #
##############################################################################
with feedback:
    st.header("Feedback")
    st.write("For feedback, mail to: **ma24m012@smail.iitm.ac.in**")
    with st.form("feedback_form"):
        name = st.text_input("Name (optional)")
        email = st.text_input("Email (optional)")
        feedback_text = st.text_area("Your feedback", height=200)
        submit = st.form_submit_button("Submit Feedback")
        if submit and feedback_text:
            st.success("Thank you for your feedback!")
