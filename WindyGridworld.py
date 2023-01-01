import GW.gridworld3 as GW3

class WindyGridworld(GW3.Gridworld):
    def __init__(self, grid_size=10) -> None:
        super().__init__(grid_size=grid_size)


if __name__ == "__main__":
    wg = WindyGridworld(grid_size=10)
    wg.simulate(T=1000)
    wg.showAnimation()