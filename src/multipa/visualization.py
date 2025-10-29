"""Functions to help visualize transcription results and
performance metrics.
Mainly collects plot creation functions to be used in jupyter notebooks.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_error_rates_by_phone_and_model(
    dataframe: pd.DataFrame,
    groupby_key: str,
    palette: str,
    hue_order: None | list[str] = None,
    xlabel: str = "Phone",
    ylabel: str = "Normalized Error Rate",
    title: str = "Error rates by phone on the Buckeye test set",
    fontsize: int = 14,
    use_confidence_intervals: bool = False,
    phone_col: str = "phone",
    err_rate_col: str = "err_rate",
    confidence_interval_col: str = "confidence_interval",
    figsize: tuple[int, int] = (25, 4),
    legend_title: str = "Experiment group/Model",
):
    """Returns a slope plot (as a line plot) showing the error rates
    of specific phones across models or experiment groups.
    Phones go along the x-axis, error rates on the y-axis and different models or model groups are
    distinguished by line color.

    Args:
        dataframe: A dataframe storing already computed error rates for phones, where each row entry
            is at minimum a (model_id, phone, err_rate)
        model_groupby_key: The column identifying the model id, name or experiment
            group used for determining line color in the plot
        palette: Name of the seaborn color palette to use
        hue_order: Optionally set a specific order of models or experiment groups to enforce
            consistency across multiple plots.
        xlabel: How to label the x-axis. Defaults to "Phone".
        ylabel: How to label the y-axis. Defaults to "Normalized Error Rate".
        title: How to title the plot. Defaults to "Error rates by phone on the Buckeye test set".
        fontsize: Defaults to 14.
        use_confidence_intervals: Set to true to display pre-computed confidence intervals for error rates
            from the 'confidence_interval_col' column in the plot. Defaults to False.
        phone_col: Column in the dataframe storing phones. Defaults to "phone".
        err_rate_col: Column in the dataframe storing error rates. Defaults to "err_rate".
        confidence_interval_col: Column in the dataframe storing pre-computed confidence
            intervals. Defaults to "confidence_interval".
        figsize: Desired matplotlib figure size. Defaults to (25, 4).
        legend_title: How to title the legend. Defaults to "Experiment group/Model".

    Returns:
        ax:  The subplot containing the line plot
    """
    group_order = dataframe.groupby(phone_col)[err_rate_col].min().sort_values()
    tmp_df = dataframe.copy(deep=True)
    tmp_df["sort_order"] = tmp_df[phone_col].map(group_order)
    tmp_df = tmp_df.sort_values("sort_order")
    if use_confidence_intervals:
        tmp_df["upper"] = tmp_df[err_rate_col] + tmp_df[confidence_interval_col]
        tmp_df["lower"] = tmp_df[err_rate_col] - tmp_df[confidence_interval_col]
        _, g = plt.subplots(figsize=figsize)
        palette = sns.color_palette(palette)
        for i, group in enumerate(hue_order):
            group_df = tmp_df[tmp_df[groupby_key] == group]
            color = palette[i]
            x = group_df[phone_col]
            g.plot(x, group_df[err_rate_col], label=group, color=color)
            g.plot(x, group_df["lower"], color=color, alpha=0.2)
            g.plot(x, group_df["upper"], color=color, alpha=0.2)
            g.fill_between(x, group_df["lower"], group_df["upper"], alpha=0.2)

    else:
        plt.figure(figsize=figsize)
        g = sns.lineplot(
            data=tmp_df, y=err_rate_col, x=phone_col, hue=groupby_key, style=groupby_key, palette=palette, hue_order=hue_order
        )

    g.set_xlabel(xlabel, fontsize=fontsize)
    g.set_ylabel(ylabel, fontsize=fontsize)
    g.set_title(title, fontsize=fontsize)
    g.tick_params(labelsize=fontsize)
    g.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=fontsize, title=legend_title, title_fontsize=fontsize)
    return g
