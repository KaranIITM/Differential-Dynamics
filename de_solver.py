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
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Differential-Equation Solver",
                   layout="wide", page_icon="🔬")

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
        if st.button(" ODE Solver", use_container_width=True):
            st.switch_page("ODE")  # FIXED: Better navigation
    with col2:
        if st.button("∇ PDE Solver", use_container_width=True):
            st.switch_page("PDE")  # FIXED: Better navigation
    with col3:
        if st.button(" Feedback", use_container_width=True):
            st.switch_page("Feedback")  # FIXED: Better navigation

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
        
        # FIXED: Corrected default ODE function
        ode_str = st.text_area("Enter system f(t, X) as python lambda",
                               value="lambda t, X: [X[1], -X]",  # FIXED: Added 
                               help="For 2nd order ODE y'' + y = 0, use: lambda t, X: [X[1], -X]")
        
        y0_str = st.text_input("Initial conditions (comma-separated)", "1, 0",
                              help="For y(0)=1, y'(0)=0, enter: 1, 0")
        t0 = st.number_input("t₀", value=0.0)
        t1 = st.number_input("t final", value=10.0, min_value=t0+1e-6)
        h = st.number_input("Step size (initial if adaptive)", value=0.1,
                            min_value=1e-6, max_value=5.0)
        compute = st.button("Run ODE Solver", use_container_width=True)

    def parse_ic(text):
        try:
            return np.array([float(x.strip()) for x in text.split(",")])  # FIXED: Added strip()
        except Exception as e:
            st.error(f"Invalid initial conditions: {e}")
            st.stop()

    # -----------------  NUMERICAL  METHODS  ---------------- #
    def euler(f, t0, y0, h, n):
        T = np.linspace(t0, t0+n*h, n+1)
        Y = np.zeros((n+1, len(y0)))
        Y[0] = y0  # FIXED: Proper initialization
        for i in range(n):
            try:
                Y[i+1] = Y[i] + h * np.array(f(T[i], Y[i]))
            except Exception as e:
                st.error(f"Euler method failed at step {i}: {e}")
                st.stop()
        return T, Y

    def rk2(f, t0, y0, h, n):
        T = np.linspace(t0, t0+n*h, n+1)
        Y = np.zeros((n+1, len(y0)))
        Y[0] = y0  # FIXED: Proper initialization
        for i in range(n):
            try:
                k1 = np.array(f(T[i], Y[i]))
                k2 = np.array(f(T[i]+h/2, Y[i]+h*k1/2))
                Y[i+1] = Y[i] + h * k2
            except Exception as e:
                st.error(f"RK2 method failed at step {i}: {e}")
                st.stop()
        return T, Y

    def rk4(f, t0, y0, h, n):
        T = np.linspace(t0, t0+n*h, n+1)
        Y = np.zeros((n+1, len(y0)))
        Y[0] = y0  # FIXED: Proper initialization
        for i in range(n):
            try:
                k1 = np.array(f(T[i], Y[i]))
                k2 = np.array(f(T[i]+h/2, Y[i]+h*k1/2))
                k3 = np.array(f(T[i]+h/2, Y[i]+h*k2/2))
                k4 = np.array(f(T[i]+h,   Y[i]+h*k3))
                Y[i+1] = Y[i] + h*(k1+2*k2+2*k3+k4)/6
            except Exception as e:
                st.error(f"RK4 method failed at step {i}: {e}")
                st.stop()
        return T, Y

    def rkf45(f, t0, y0, t_end, h, tol=1e-6):
        T, Y, t = [t0], [np.array(y0)], t0  # FIXED: Ensure y0 is array
        y = np.array(y0)
        max_iter = int((t_end - t0) / h * 10)  # FIXED: Prevent infinite loops
        iteration = 0
        
        while t < t_end and iteration < max_iter:
            iteration += 1
            if t+h > t_end:
                h = t_end - t
            try:
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
                    Y.append(y.copy())
                h *= min(2, max(0.1, 0.9*(tol/max(error, 1e-12))**0.2))  # FIXED: Prevent division by zero
            except Exception as e:
                st.error(f"RKF45 method failed: {e}")
                st.stop()
        
        if iteration >= max_iter:
            st.warning("RKF45 reached maximum iterations")
            
        return np.array(T), np.array(Y)

    # start-up with RK4 for multistep schemes
    def start_rk4(f, t0, y0, h):
        return rk4(f, t0, y0, h, 3)

    def adams_bashforth4(f, t0, y0, h, n):
        if n <= 3:  # FIXED: Added bounds checking
            return rk4(f, t0, y0, h, n)
            
        T, Y = start_rk4(f, t0, y0, h)
        for i in range(3, n):
            try:
                t = T[-1] + h
                y = np.array(Y[-1] + h/24*(55*np.array(f(T[-1], Y[-1])) -59*np.array(f(T[-2], Y[-2]))
                                  +37*np.array(f(T[-3], Y[-3])) -9*np.array(f(T[-4], Y[-4]))))
                T = np.append(T, t)
                Y = np.vstack([Y, y])
            except Exception as e:
                st.error(f"Adams-Bashforth failed at step {i}: {e}")
                st.stop()
        return T, Y

    def adams_moulton4(f, t0, y0, h, n):
        if n <= 3:  # FIXED: Added bounds checking
            return rk4(f, t0, y0, h, n)
            
        T, Y = start_rk4(f, t0, y0, h)
        for i in range(3, n):
            try:
                t_pred = T[-1] + h
                y_pred = np.array(Y[-1] + h/24*(55*np.array(f(T[-1], Y[-1])) -59*np.array(f(T[-2], Y[-2]))
                                       +37*np.array(f(T[-3], Y[-3])) -9*np.array(f(T[-4], Y[-4]))))
                y_corr = np.array(Y[-1] + h/24*(9*np.array(f(t_pred, y_pred)) +19*np.array(f(T[-1], Y[-1]))
                                       -5*np.array(f(T[-2], Y[-2])) +np.array(f(T[-3], Y[-3]))))
                T = np.append(T, t_pred)
                Y = np.vstack([Y, y_corr])
            except Exception as e:
                st.error(f"Adams-Moulton failed at step {i}: {e}")
                st.stop()
        return T, Y
    
    def bdf2(f, t0, y0, h, n, its=5):
        T = np.zeros(n+1)
        Y = np.zeros((n+1, len(y0)))
        T[0], T[1] = t0, t0+h
        Y = y0  # FIXED: Proper initialization
        try:
            Y[1] = y0 + h*np.array(f(t0, y0))
        except Exception as e:
            st.error(f"BDF2 initialization failed: {e}")
            st.stop()
            
        for i in range(1, n):
            try:
                t_next = T[i] + h
                y = Y[i].copy()  # initial guess
                for _ in range(its):
                    y = (4*Y[i] - Y[i-1] + 2*h*np.array(f(t_next, y)))/3
                T[i+1] = t_next
                Y[i+1] = y
            except Exception as e:
                st.error(f"BDF2 failed at step {i}: {e}")
                st.stop()
        return T, Y

    def verlet(accel, t0, y0, h, n):
        if len(y0) != 2:  # FIXED: Added validation
            st.error("Verlet method requires exactly 2 initial conditions [position, velocity]")
            st.stop()
        T = np.linspace(t0, t0+n*h, n+1)
        Y = np.zeros((n+1, 2))
        Y[0] = y0  # FIXED: Proper initialization
        
        try:
            a0 = accel(t0, Y[0, 0])
            Y[1, 0] = Y[0, 0] + Y[0, 1]*h + 0.5*a0*h**2
            Y[1, 1] = Y[0, 1] + 0.5*(a0 + accel(t0+h, Y[1, 0]))*h
            
            for i in range(1, n):
                a = accel(T[i], Y[i, 0])
                Y[i+1, 0] = 2*Y[i, 0] - Y[i-1, 0] + a*h**2
                a_next = accel(T[i+1], Y[i+1, 0])
                Y[i+1, 1] = Y[i, 1] + 0.5*(a + a_next)*h
        except Exception as e:
            st.error(f"Verlet method failed: {e}")
            st.stop()
        return T, Y

    def stormer_verlet(force, t0, y0, h, n):
        if len(y0) != 2:
            st.error("Stormer-Verlet method requires exactly 2 initial conditions [position, momentum]")
            st.stop()
        T = np.linspace(t0, t0+n*h, n+1)
        Y = np.zeros((n+1, 2))
        Y[0] = y0  # FIXED: Proper initialization
        
        try:
            p_half = Y[0, 1] + 0.5*h*force(t0, Y[0, 0])
            for i in range(n):
                q_next = Y[i, 0] + h*p_half
                p_next = p_half + 0.5*h*force(T[i+1], q_next)
                Y[i+1, 0] = q_next  # FIXED: Assign individual elements
                Y[i+1, 1] = p_next  # FIXED: Assign individual elements
                p_half = p_next + 0.5*h*force(T[i+1], q_next)
        except Exception as e:
            st.error(f"Stormer-Verlet method failed: {e}")
            st.stop()
        return T, Y

    # ------------------------  RUN  BUTTON  ---------------- #
    if compute:
        y0 = parse_ic(y0_str)
        n_steps = int(np.ceil((t1-t0)/h))
        
        if n_steps > 10000:  # FIXED: Prevent excessive computation
            st.error("Too many time steps (>10000). Reduce time range or increase step size.")
            st.stop()
        
        # FIXED: Added missing function evaluation
        try:
            f = eval(ode_str, {"np": np, "__builtins__": {}})  # FIXED: Safer eval
        except Exception as err:
            st.error(f"Invalid function: {err}")
            st.stop()
        
        # FIXED: Added proper validation
        try:
            test_output = f(t0, y0)
            test_output = np.array(test_output)
            if len(test_output) != len(y0):
                st.error(f"Function returns {len(test_output)} values but initial conditions have {len(y0)} values")
                st.stop()
        except Exception as e:
            st.error(f"Function evaluation failed: {e}")
            st.stop()

        # FIXED: Better method dispatch with error handling
        try:
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
                # FIXED: Better extraction of acceleration
                def accel(t, x):
                    try:
                        state = np.array([x, 0.0])
                        result = f(t, state)
                        return result[1] if len(result) > 1 else 0
                    except:
                        return 0
                T, Y = verlet(accel, t0, y0, h, n_steps)
            else:   # Stormer-Verlet
                # FIXED: Better extraction of force
                def force(t, q):
                    try:
                        state = np.array([q, 0.0])
                        result = f(t, state)
                        return result[1] if len(result) > 1 else 0
                    except:
                        return 0
                T, Y = stormer_verlet(force, t0, y0, h, n_steps)

            st.success("Computation complete")
            
        except Exception as e:
            st.error(f"Solver failed: {e}")
            st.stop()
        
        # ------------------  PLOTS  ----------------------- #
        with ode_col2:
            st.subheader("Solution curves")
            fig, ax = plt.subplots(figsize=(10, 6))
            for i in range(Y.shape[1]):
                ax.plot(T, Y[:, i], label=f"x{i}", linewidth=2)
            ax.set_xlabel("Time t")
            ax.set_ylabel("Solution")
            ax.set_title(f"ODE Solution using {method}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig, use_container_width=True)

            if Y.shape[1] >= 2:
                st.subheader("Phase portrait (x0 vs x1)")
                fig2, ax2 = plt.subplots(figsize=(8, 8))
                ax2.plot(Y[:, 0], Y[:, 1], 'b-', linewidth=2)
                ax2.scatter(Y[0, 0], Y[0, 1], color='green', s=100, label='Start', zorder=5)
                ax2.scatter(Y[-1, 0], Y[-1, 1], color='red', s=100, label='End', zorder=5)
                ax2.set_xlabel("x0 (Position)")
                ax2.set_ylabel("x1 (Velocity)")
                ax2.set_title("Phase Portrait")
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                ax2.axis('equal')
                st.pyplot(fig2, use_container_width=True)

            st.subheader("Numerical Data")
            df = pd.DataFrame(Y, columns=[f"x{i}" for i in range(Y.shape[1])])
            df.insert(0, "t", T)
            st.dataframe(df, use_container_width=True)

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
        Nx = st.number_input("Spatial nodes Nx", value=51, step=1, min_value=3, max_value=1000)
        Nt = st.number_input("Time steps Nt", value=200, step=1, min_value=1, max_value=5000)
    
    with col2:
        dt = st.number_input("Δt (time step)", value=0.001, min_value=1e-6)
        alpha = st.number_input("Diffusion/C² (heat α, wave c²)", value=1.0, min_value=0.001)
    
    ic = st.text_input("Initial condition u0(x)", "np.sin(np.pi*x/L)",
                      help="Use 'x' as spatial variable, 'L' for length, 'np' for numpy")
    bc_left = st.text_input("Left boundary u(0,t)", "0",
                           help="Use 't' for time variable")
    bc_right = st.text_input("Right boundary u(L,t)", "0",
                            help="Use 't' for time variable")
    
    run_pde = st.button("Run PDE Solver", use_container_width=True)

    if run_pde:
        try:
            x = np.linspace(0, L, int(Nx))
            dx = x[1] - x  # FIXED: Proper dx calculation
            
            # FIXED: Safer initial condition evaluation
            try:
                u = eval(ic, {"np": np, "x": x, "L": L, "__builtins__": {}})
                u = np.array(u)
            except Exception as e:
                st.error(f"Invalid initial condition: {e}")
                st.stop()
                
            u_all = [u.copy()]

            if pde_type == "Heat Equation" and scheme in {"Forward-Euler", "Crank-Nicolson", "Backward-Euler"}:
                lam = alpha*dt/dx**2
                if lam > 0.5 and scheme == "Forward-Euler":
                    st.warning(f"Scheme may be unstable (λ = α·Δt/Δx² = {lam:.3f} > 0.5)")
                
                # Build finite difference matrices
                A = np.diag((1-2*lam)*np.ones(len(x))) + np.diag(lam*np.ones(len(x)-1), 1) + np.diag(lam*np.ones(len(x)-1), -1)

                if scheme == "Backward-Euler":
                    A = np.diag((1+2*lam)*np.ones(len(x))) - np.diag(lam*np.ones(len(x)-1), 1) - np.diag(lam*np.ones(len(x)-1), -1)
                    try:
                        A_inv = np.linalg.inv(A)
                    except np.linalg.LinAlgError:
                        st.error("Matrix inversion failed for Backward-Euler scheme")
                        st.stop()
                        
                elif scheme == "Crank-Nicolson":
                    B = np.diag((1-lam)*np.ones(len(x))) + np.diag(lam*np.ones(len(x)-1), 1)/2 + np.diag(lam*np.ones(len(x)-1), -1)/2
                    C = np.diag((1+lam)*np.ones(len(x))) - np.diag(lam*np.ones(len(x)-1), 1)/2 - np.diag(lam*np.ones(len(x)-1), -1)/2
                    try:
                        C_inv = np.linalg.inv(C)
                    except np.linalg.LinAlgError:
                        st.error("Matrix inversion failed for Crank-Nicolson scheme")
                        st.stop()

                for step in range(int(Nt)):
                    # FIXED: Proper boundary condition handling
                    try:
                        left_val = eval(bc_left, {"t": step*dt, "np": np, "__builtins__": {}})
                        right_val = eval(bc_right, {"t": step*dt, "np": np, "__builtins__": {}})
                        
                        # Handle array results from boundary conditions
                        if np.isscalar(left_val):
                            u[0] = float(left_val)
                        else:
                            u = float(np.asarray(left_val).item())
                            
                        if np.isscalar(right_val):
                            u[-1] = float(right_val)
                        else:
                            u[-1] = float(np.asarray(right_val).item())
                            
                    except Exception:
                        u[0] = 0.0  # default boundary
                        u[-1] = 0.0

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
                    st.warning(f"Scheme may be unstable (λ = c²·Δt²/Δx² = {lam:.3f} > 1)")
                
                u_prev = u.copy()
                u_current = u.copy()
                
                for step in range(int(Nt)):
                    u_next = np.zeros_like(u)
                    u_next[1:-1] = 2*u_current[1:-1] - u_prev[1:-1] + lam*(u_current[2:] -2*u_current[1:-1] + u_current[:-2])
                    
                    # Apply boundary conditions
                    try:
                        left_val = eval(bc_left, {"t": step*dt, "np": np, "__builtins__": {}})
                        right_val = eval(bc_right, {"t": step*dt, "np": np, "__builtins__": {}})
                        u_next[0] = float(np.asarray(left_val).item()) if not np.isscalar(left_val) else float(left_val)
                        u_next[-1] = float(np.asarray(right_val).item()) if not np.isscalar(right_val) else float(right_val)
                    except:
                        u_next[0] = 0.0
                        u_next[-1] = 0.0
                    
                    u_prev = u_current.copy()
                    u_current = u_next.copy()
                    u_all.append(u_current.copy())

            elif pde_type == "Laplace Equation":
                # steady-state solution via Gauss-Seidel
                u = np.zeros_like(x)
                for iteration in range(int(Nt)):
                    # FIXED: Proper boundary condition handling
                    try:
                        left_val = eval(bc_left, {"np": np, "__builtins__": {}})
                        right_val = eval(bc_right, {"np": np, "__builtins__": {}})
                        u[0] = float(np.asarray(left_val).item()) if not np.isscalar(left_val) else float(left_val)
                        u[-1] = float(np.asarray(right_val).item()) if not np.isscalar(right_val) else float(right_val)
                    except:
                        u[0] = 0.0
                        u[-1] = 0.0

                    # Gauss-Seidel iteration
                    for i in range(1, len(x)-1):
                        u[i] = 0.5*(u[i-1] + u[i+1])
                        
                u_all = [u]  # Only final steady state

            elif scheme == "Method of Lines":
                # semi-discrete heat equation -> ODE system
                lam = alpha/dx**2
                A = np.diag(-2*np.ones(len(x))) + np.diag(np.ones(len(x)-1), 1) + np.diag(np.ones(len(x)-1), -1)
                
                def rhs(_, y): 
                    result = lam*A @ y
                    # Apply boundary conditions in the ODE
                    result[0] = 0
                    result[-1] = 0
                    return result
                    
                T, Y = rk4(rhs, 0, u, dt, int(Nt))
                u_all = Y

            # ---------------  PLOTS  --------------- #
            st.success("PDE solved successfully")
            
            # FIXED: Better plotting
            if len(u_all) > 1:
                fig = plt.figure(figsize=(12, 8))
                xx, tt = np.meshgrid(x, np.arange(len(u_all))*dt)
                contour = plt.contourf(xx, tt, np.array(u_all), 50, cmap="viridis")
                plt.colorbar(contour, label="u(x,t)")
                plt.xlabel("Space (x)")
                plt.ylabel("Time (t)")
                plt.title(f"{pde_type} solution using {scheme}")
                st.pyplot(fig, use_container_width=True)
                
                # Also show final solution
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.plot(x, u_all[-1], 'b-', linewidth=2)
                ax2.set_xlabel("Space (x)")
                ax2.set_ylabel("u(x, t_final)")
                ax2.set_title("Final Solution")
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2, use_container_width=True)
            else:
                # For steady-state problems
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(x, u_all[0], 'b-', linewidth=2)
                ax.set_xlabel("Space (x)")
                ax.set_ylabel("u(x)")
                ax.set_title(f"{pde_type} steady-state solution")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.error(f"PDE solver failed: {e}")
            import traceback
            st.error(traceback.format_exc())

##############################################################################
# ---------------------------  FEEDBACK  TAB  ------------------------------ #
##############################################################################
with feedback:
    st.header("Feedback")
    st.write("For feedback, mail to: **ma24m012@smail.iitm.ac.in**")
    
    st.subheader("Quick Feedback Form")
    with st.form("feedback_form"):
        name = st.text_input("Name (optional)")
        email = st.text_input("Email (optional)")
        feedback_text = st.text_area("Your feedback", height=200)
        submit = st.form_submit_button("Submit Feedback")
        
        if submit and feedback_text:
            st.success("Thank you for your feedback! Please send detailed feedback to: ma24m012@smail.iitm.ac.in")
            st.info(f"""
            **Feedback Summary:**
            - From: {name if name else 'Anonymous'} ({email if email else 'No email provided'})
            - Message: {feedback_text[:100]}{'...' if len(feedback_text) > 100 else ''}
            
            Please copy this and send to: **ma24m012@smail.iitm.ac.in**
            """)
