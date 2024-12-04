import matplotlib.pyplot as plt


def plot_tsp_solutions(
    locations,
    solution1,
    solution2,
    solution3,
    title1="TSP Solution 1",
    title2="TSP Solution 2",
    title3="TSP Solution 3",
    distance1=None,
    distance2=None,
    distance3=None,
    action=None,
):
    """
    Plot three TSP solutions on three subplots.

    :param locations: Dictionary with keys as node indices and values as coordinates.
    :param solution1: List of integers representing the first TSP solution.
    :param solution2: List of integers representing the second TSP solution.
    :param solution3: List of integers representing the third TSP solution.
    :param action: List of two integers representing the nodes to be colored
    differently.
    """
    # Convert dictionary to list of tuples
    locations_list = [locations[i] for i in range(len(locations))]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot first solution if not None
    if solution1 is not None:
        axs[0].set_title(title1 + f" ({distance1})")
        for i in range(len(solution1)):
            start = locations[solution1[i - 1]]
            end = locations[solution1[i]]
            axs[0].plot([start[0], end[0]], [start[1], end[1]], "bo-")
        axs[0].scatter(*zip(*locations_list), c="red")
        if action is not None:
            axs[0].scatter(*locations[action[0]], c="yellow", s=100, label="Action 0")
            axs[0].scatter(*locations[action[1]], c="red", s=100, label="Action 1")

    # Plot second solution if not None
    if solution2 is not None:
        axs[1].set_title(title2 + f" ({distance2})")
        for i in range(len(solution2)):
            start = locations[solution2[i - 1]]
            end = locations[solution2[i]]
            axs[1].plot([start[0], end[0]], [start[1], end[1]], "bo-")
        axs[1].scatter(*zip(*locations_list), c="red")
        if action is not None:
            axs[1].scatter(*locations[action[0]], c="yellow", s=100, label="Action 0")
            axs[1].scatter(*locations[action[1]], c="red", s=100, label="Action 1")

    # Plot third solution if not None
    if solution3 is not None:
        axs[2].set_title(title3 + f" ({distance3})")
        for i in range(len(solution3)):
            start = locations[solution3[i - 1]]
            end = locations[solution3[i]]
            axs[2].plot([start[0], end[0]], [start[1], end[1]], "bo-")
        axs[2].scatter(*zip(*locations_list), c="red")
        if action is not None:
            axs[2].scatter(*locations[action[0]], c="yellow", s=100, label="Action 0")
            axs[2].scatter(*locations[action[1]], c="red", s=100, label="Action 1")

    plt.show()
