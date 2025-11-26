import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np

class Dashboard: 
    def make_eeg_fig(raw_eeg):
        sfreq = raw_eeg["sfreq"]
        eeg = raw_eeg["data"]
        # downsample for display (1000 timepoints max)
        ds = max(1, eeg.shape[1] // 1000)
        eeg = eeg[:, ::ds]

        eeg_z = (eeg - eeg.mean(axis=1, keepdims=True)) / (eeg.std(axis=1, keepdims=True) + 1e-12)
        eeg_z = np.asarray(eeg_z, dtype=float)

        fig = go.Figure(go.Heatmap(
            z=eeg_z,
            colorscale="Turbo"
        ))
        fig.update_layout(title="EEG heatmap", height=300)
        print(np.nanmin(raw_eeg["data"]), np.nanmax(raw_eeg["data"]))
        print("EEG shape:", eeg.shape)
        print("EEG mean/std:", np.nanmean(eeg), np.nanstd(eeg))
        print("Z min/max:", np.nanmin(eeg_z), np.nanmax(eeg_z))
        return fig

    def make_emg_lines(raw_emg):
        sfreq = raw_emg["sfreq"]
        emg = raw_emg["data"]
        emg_ch = raw_emg["ch_names"]
        t = np.arange(emg.shape[1]) / sfreq
        fig = go.Figure()
        for i,ch in enumerate(emg_ch):
            fig.add_trace(go.Scatter(x=t, y=emg[i] + i*200, name=ch, mode="lines"))
        fig.update_layout(title="EMG (stacked)", height=300)
        return fig

    def make_imu_figs(imu_vec, imu_kal=None):
        fig = go.Figure()

        # Always add complementary traces
        for sensor, dfv in imu_vec.items():
            t = dfv["t_sec"]

            for axis in ["pitch", "roll", "yaw"]:
                if axis in dfv.columns:
                    fig.add_trace(go.Scatter(
                        x=t, y=dfv[axis],
                        name=f"{sensor} {axis} (vec)"
                    ))

        # Add Kalman if given
        if imu_kal is not None:
            for sensor, dfk in imu_kal.items():
                if "pitch" in dfk.columns:
                    fig.add_trace(go.Scatter(
                        x=dfk["t_sec"], y=dfk["pitch"],
                        name=f"{sensor} pitch (kal)", opacity=0.6
                    ))

        fig.update_layout(title="IMU: pitch (vec" + (" + kal)" if imu_kal else ")"),
                        height=350)
        return fig
    @staticmethod
    def build_dashboard_all_subjects(all_subjects_data):
        subject_ids = sorted(list(all_subjects_data.keys()))
        app = dash.Dash(__name__)
        app.layout = html.Div([
            html.H2("Multimodal Viewer"),
            dcc.Dropdown(id="subject-dropdown", options=[{"label":s,"value":s} for s in subject_ids],
                        value=subject_ids[0]),
            dcc.Graph(id="eeg-graph"),
            dcc.Graph(id="emg-graph"),
            dcc.Graph(id="imu-graph"),
            html.Div(id="meta")
        ], style={"width":"95%", "margin":"auto"})

        @app.callback(
            Output("eeg-graph", "figure"),
            Output("emg-graph", "figure"),
            Output("imu-graph", "figure"),
            Output("meta", "children"),
            Input("subject-dropdown", "value")
        )
        def update_plots(subj_id):
            sd = all_subjects_data[subj_id]
            raw_eeg = sd["raw_eeg"]
            raw_emg = sd["raw_emg_filtered"]

            imu_vec = sd.get("imu_fused_vec", {})
            imu_kal = sd.get("imu_fused_kal", None)   # will be None if disabled

            eeg_fig = Dashboard.make_eeg_fig(raw_eeg) if raw_eeg is not None else go.Figure()
            emg_fig = Dashboard.make_emg_lines(raw_emg) if raw_emg is not None else go.Figure()
            #If kalman is included
            imu_fig = Dashboard.make_imu_figs(imu_vec, imu_kal) 

            meta_txt = f"Subject {subj_id} — events: {len(sd.get('events', []))}"

            return eeg_fig, emg_fig, imu_fig, meta_txt

        return app
