################################################################################
# File: read_gpickle copy.py
# Purpose:
#   Compatibility shim that exposes the legacy tracking-oriented API using the
#   canonical merged implementation in read_gpickle.py.
################################################################################
from legacy_shared_utils import get_available_species, network_avg, network_radius
from read_gpickle import (
    draw_species_image_tracking as draw_species_image,
    draw_state_image,
    find_cell_in_network,
    find_cell_in_network_by_id,
    plot_tracked_cells_data,
    read_network_tracking as read_network,
    run_tracking_cli,
    select_cell_to_track,
    select_cells_by_index,
    select_cells_by_position,
)


if __name__ == '__main__':
    path = '/Users/huyunchen/Documents/PhD/test/1.0/'
    run_tracking_cli(path, t0=0, tf=600)
