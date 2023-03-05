import shap
from shap import Explanation

from matplotlib import pyplot as plt


labels = {
    "VALUE": "Feature Value",
    "FEATURE": "Feature",
    "FEATURE_VALUE": "Feature Value",
    "INTERACTION_VALUE": "Interaction Value"
}


def patched_beeswarm(
        shap_values, max_display=10, order=Explanation.abs.mean(0),
        clustering=None, cluster_threshold=0.5, color=None,
        axis_color="#333333", alpha=1, log_scale=False,
        color_bar=True, plot_size="auto",
        color_bar_label=labels["FEATURE_VALUE"]
        ):

    """Simple patch of SHAP made to return the figure if show == False"""

    plt.ioff()

    fig = shap.plots.beeswarm(
        shap_values, max_display=max_display, order=order,
        clustering=clustering, cluster_threshold=cluster_threshold,
        color=color, axis_color=axis_color, alpha=alpha, show=False,
        log_scale=log_scale, color_bar=color_bar, plot_size=plot_size,
        color_bar_label=color_bar_label
    )

    # Return the resulting figure
    return fig
