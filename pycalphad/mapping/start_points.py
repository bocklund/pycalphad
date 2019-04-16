import numpy as np
from copy import deepcopy
from .utils import convex_hull, sort_x_by_y, opposite_direction, v_array, Direction, find_two_phase_region_compsets, get_compsets
from .compsets import BinaryCompSet
from pycalphad import variables as v, calculate, equilibrium


class StartPoint():
    def __init__(self, temperature, direction, compsets, composition=None):
        self.temperature = temperature
        self.direction = direction
        self.compsets = compsets
        if composition is not None:
            self.composition = composition
        else:
            # get the average composition from the compsets
            self.composition = BinaryCompSet.mean_composition(compsets)

    def __repr__(self):
        phases = "/".join([c.phase_name for c in self.compsets])
        if self.direction is Direction.POSITIVE:
            dir_str = "+"
        else:
            dir_str = "-"
        return "<StartPoint T={:0.2f}, dT=({}), X={:0.3f}, Phases={}>".format(
            self.temperature, dir_str, self.composition, phases)

    def isduplicate(self, other, comp_tol=0.01, temp_tol=1):
        """
        Check for equality between two StartPoints.

        Parameters
        ----------
        other : StartPoint

        Returns
        -------
        bool

        Notes
        -----
        Two StartPoints are equal if they are the same length and all the
        overall compositions are equal, and if they go the same direction.
        The order of the composition sets has to also be the same, such that
        two StartPoints that are on either side of a congruent (for example)
        are not equivalent.
        """
        if self.direction == other.direction and len(self.compsets) == len(other.compsets):
            dup = all([any([c.isclose(o, comp_tol=comp_tol, temp_tol=temp_tol) for o in other.compsets]) for c in self.compsets])
            self_phases = [c.phase_name for c in BinaryCompSet.composition_sorted(self.compsets)]
            other_phases = [c.phase_name for c in BinaryCompSet.composition_sorted(other.compsets)]
            order_is_same = self_phases == other_phases
            return dup and order_is_same
        else:
            return False


class StartPointsList():
    def __init__(self, eq_comp_tol=0.01, eq_temp_tol=1):
        self.visited_start_points = []
        self.remaining_start_points = []
        self.eq_comp_tol = eq_comp_tol
        self.eq_temp_tol = eq_temp_tol

    def __repr__(self):
        pts_str = ", ".join([repr(p) for p in self.remaining_start_points])
        return "[" + pts_str + "]"

    def contains_start_point(self, start_point):
        """
        Return True if the start_point is already in the list of all start points.

        Parameters
        ----------
        start_point : StartPoint

        Returns
        -------
        bool

        """
        in_visited = any([start_point.isduplicate(sp, self.eq_comp_tol, self.eq_temp_tol) for sp in self.visited_start_points])
        in_remanining = any([start_point.isduplicate(sp, self.eq_comp_tol, self.eq_temp_tol) for sp in self.remaining_start_points])
        return in_visited or in_remanining

    def visited_start_point(self, start_point):
        """
        Return True if the start_point has already been visited.

        Parameters
        ----------
        start_point : StartPoint

        Returns
        -------
        bool

        """
        return any([start_point.isduplicate(sp, self.eq_comp_tol, self.eq_temp_tol) for sp in self.visited_start_points])

    def add_start_point(self, start_point, add_duplicates=False):
        """
        Add a start point

        Parameters
        ----------
        start_point : StartPoint
        add_duplicates : bool
            Whether duplicate StartPoints can be added. Defaults to False.

        Returns
        -------
        bool
            True if a start point was added, False otherwise
        """
        if add_duplicates or not self.contains_start_point(start_point):
            self.remaining_start_points.append(start_point)
            return True
        return False


    def add_end_point(self, end_point):
        """
        Add a "start point" in the opposite direction of a current

        Parameters
        ----------
        end_point : StartPoint
        """
        if not self.contains_start_point(end_point):
            self.visited_start_points.append(end_point)

    def get_next_start_point(self,):
        """
        Return the next start point

        Returns
        -------
        StartPoint

        """
        while len(self.remaining_start_points) > 0:
            candidate_start_point = self.remaining_start_points.pop(0)
            if not self.visited_start_point(candidate_start_point):
                self.visited_start_points.append(candidate_start_point)
                return candidate_start_point
        else:
            return None


def find_three_phase_start_points(new_compsets, prev_compsets, direction):
    """
    Returns two new start points from a three phase invariant reaction

    Notes
    -----
    This is around a three phase equilibrium invariant reaction.

    There are two situations:
    1. Two two-phase regions above, one two-phase region below (eutectic)
    2. One two-phase region above, two two-phase regions below (peritectic)

    Graphically, an invariant reaction occurs at `=`

    Situation 1 (eutectic-like)
    |------------------|
    |------------------|
    |------------------|
    |------------------|
    |========|=========|
    |--------|---------|
    |--------|---------|
    |--------|---------|
    |--------|---------|

    Situation 2 (peritectic-like)
    |--------|---------|
    |--------|---------|
    |--------|---------|
    |--------|---------|
    |--------|---------|
    |========|=========|
    |------------------|
    |------------------|
    |------------------|
    |------------------|

    We need to find all three regions, remove the one we have mapped already
    and add the two new ones. The best way to do this would be to
    calculate three phase equilibrium, since we cannot do this
    in pycalphad currently, we have to approximate it.

    We exploit the fact that we are always transferring between a large and small region.
    Thus we add the start point for the new region we found (in the same direction) and we
    always have to find the opposing small region. If we go from a large to small region,
    the opposing small region should be in the same direction, otherwise if we go from a small
    to large region, the new small region direction should have the opposite direction.

    We also reassign the temperatures so that the next step (T+delta) will add
    to the composition grid correctly.
    """
    prev_phases = [c.phase_name for c in prev_compsets]
    prev_comps = [c.composition for c in prev_compsets]
    prev_comps_diff = np.abs(np.max(prev_comps) - np.min(prev_comps))
    prev_temperature = prev_compsets[0].temperature

    new_phases = [c.phase_name for c in new_compsets]
    new_comps = [c.composition for c in new_compsets]
    new_comps_diff = np.abs(np.max(new_comps) - np.min(new_comps))
    new_temperature = new_compsets[0].temperature

    # In all cases, we want a new StartPoint for the new compsets in the direction we were going
    start_points = [StartPoint(prev_temperature, direction, new_compsets)]

    # assign small and large regions
    if (new_comps_diff < prev_comps_diff):  # went from large to small region
        L_cs = prev_compsets  # large region
        S_cs = new_compsets  # small region
        L_phases = prev_phases
        S_phases = new_phases
        new_direction = direction
        opp_reg_temperature = prev_temperature
    else:  # went from small to large region
        L_cs = new_compsets  # large region
        S_cs = prev_compsets  # small region
        L_phases = new_phases
        S_phases = prev_phases
        new_direction = opposite_direction(direction)
        opp_reg_temperature = new_temperature

    opposing_small_region_cs = [c for c in S_cs if c.phase_name not in L_phases] + [c for c in L_cs if c.phase_name not in S_phases]
    start_points.append(StartPoint(opp_reg_temperature, new_direction, opposing_small_region_cs))

    return start_points


def find_nearby_region_start_point(dbf, comps ,phases, compsets, indep_comp_idx, temperature, dT,
                                   conds, indep_comp_cond, start_point_list, cutoff_search_distance=0.1,
                                   verbose=False, graceful=True, hull_kwargs=None):
    """
    Return a starting point for a nearby region.

    Parameters
    ----------
    dbf : pycalphad.Database

    compsets : list
    cutoff_search_distance : float
        Distance in composition to cutoff the search for new phases.
    start_point_list : StartPointsList

    The idea here is that the compsets have converged to each other (e.g. at a congruent melting point)
    and we've mapped out one side of the point and need to find the other side.

    The idea is that we select several temperatures and construct a convex hull in composition
    at those temperatures to search the composition region. Then we will go through the points in overall composition
    from nearest to farthest from the average composition and try to find where there is
    1. Two phases in equilibrium
    2. At least one common phase with the current equilibrium
    3. The ordering of the phases w.r.t composition are different e.g. (X(LIQUID)>X(CU2MG) in one set vs. X(LIQUID)<X(CU2MG) in another set)
    for the positive and negative directions.
    """
    str_comp = str(indep_comp_cond.species.name)
    average_comp = BinaryCompSet.mean_composition(compsets)
    sorted_phases = [c.phase_name for c in BinaryCompSet.composition_sorted(compsets)]  # phases sorted by min to max composition
    current_phases_set = set(sorted_phases)

    # first we'll search temperatures very close to the current temperature (shifted by dT/10)
    trial_Ts = [
        (temperature - dT / 10.0, Direction.NEGATIVE),
        (temperature + dT / 10.0, Direction.POSITIVE),
    ]

    # take the first result we get
    for trial_T, trial_direction in trial_Ts:
        conds[v.T] = trial_T
        conds[indep_comp_cond] = v_array(average_comp, cutoff_search_distance, 0.005)
        hull = convex_hull(dbf, comps, phases, conds, **hull_kwargs)

        out_phases, compositions, site_fracs = hull[1], hull[3], hull[4]
        grid_shape = out_phases.shape[:-1]
        num_phases = out_phases.shape[-1]
        it = np.nditer(np.empty(grid_shape), flags=['multi_index'])  # empty grid for indexing
        while not it.finished:
            idx = it.multi_index
            trial_compsets = []
            for i in np.arange(num_phases):
                compset = BinaryCompSet(str(out_phases[idx][i]), temperature, str_comp, compositions[idx][i, indep_comp_idx], site_fracs[idx][i, :])
                trial_compsets.append(compset)
            trial_phases = [c.phase_name for c in trial_compsets]
            trial_phases_set = set(trial_phases)
            trial_compositions = [c.composition for c in trial_compsets]
            sorted_trial_phases = sort_x_by_y(trial_phases, trial_compositions)
            # Convex hull always gives back pairs of compsets, even for true single phase regions.
            # We need to filter out regions where the phases aren't the same, those aren't true two phase regions.
            # This might break in a miscibility gap.
            # Condition 1: Number of phases must be 2
            if len(trial_phases_set) != 2:
                it.iternext()
                continue
            # Condition 2: Must share one unique phase
            if len(current_phases_set.intersection(trial_phases_set)) < 1:
                it.iternext()
                continue
            # Condition 3: Ordering of the set of phases must be different
            if sorted_phases == sorted_trial_phases:
                it.iternext()
                continue
            # If we made it here, we found a potential match!
            sp = StartPoint(trial_T - trial_direction*dT, trial_direction, trial_compsets)
            if start_point_list.add_start_point(sp):
                return sp # We found a valid start point
            else:
                it.iternext()
                continue  # We didn't find a valid start point, keep going.
            # Don't add boundaries because this is an inaccurate set
    if graceful:
        return
    else:
        raise ValueError( "Could not find start point for neighbor to compsets: {}".format(compsets))


def find_X_start_point(dbf, comps, phases, conditions, T, indep_comp, indep_comp_idx, dT, max_discrepancy=0.01, verbosity=0, hull_kwargs=None):
    """Find a StartPoint by searching in composition"""
    hull_kwargs = hull_kwargs or dict()
    found_nodes = []
    curr_conds = deepcopy(conditions)
    curr_conds[v.T] = T
    x_cond = [k for k in conditions.keys() if isinstance(k, v.X)][0]
    hull = convex_hull(dbf, comps, phases, curr_conds, **hull_kwargs)
    cs = find_two_phase_region_compsets(hull, T, indep_comp, indep_comp_idx, discrepancy_tol=max_discrepancy)
    if len(cs) == 2:
        # verify that these show up in the equilibrium calculation
        specific_conds = deepcopy(curr_conds)
        specific_conds[x_cond] = BinaryCompSet.mean_composition(cs)
        eq_cs = get_compsets(equilibrium(dbf, comps, phases, specific_conds, **hull_kwargs), indep_comp=indep_comp, indep_comp_index=indep_comp_idx)
        if len(eq_cs) == 2:
            # add a direction of dT > 0 and dT < 0
            # shift starting_T so they start at the same place.
            found_nodes.append(StartPoint(T - dT, Direction.POSITIVE, eq_cs))
            found_nodes.append(StartPoint(T + dT, Direction.NEGATIVE, eq_cs))
    return found_nodes


def find_T_start_point(dbf, comps, phases, conditions, pure_component, X_distance=0.05, num_X=40, hull_kwargs=None):
    """Find a StartPoint by searching in temperature, special cased for a pure element

    Parameters
    ----------
    dbf : Database
    comps : list
        Components to consider.
    phases : list
        Phases to consider.
    pure_component : str
        Pure component (must be in comps) to
    T : Tuple
        Temperature grid for calculation. Either a tuple of (T_min, T_max, T_step) or a NumPy array of a temperature grid (must be understood by pycalphad)
    P : float
        Scalar pressure to use in calculation.
    X_distance : float, optional
        Distance from the edge of composition space to search for a two phase node.
    verbosity : int, optional

    """
    # TODO: Binary assumption, multicomponent generalization needs equilibrium vs. T
    hull_kwargs = hull_kwargs or dict()
    # find which component is the degree of freedom (matters for composition set ordering)

    comp_keys = [key for key, val in conditions.items() if isinstance(key, v.X)]
    if len(comp_keys) == 1:
        comp_key = comp_keys[0]
        dof_comp = comp_key.name[2:]
    else:
        raise ValueError("Too many component degrees of freedom")
    indep_comp_idx = sorted(comps).index(dof_comp)

    T_cond = conditions[v.T]
    dT = T_cond[2]  # used when generating nodes
    calc_T_cond = (T_cond[0], T_cond[1], 1)

    # calculate the pure element as a function of temperature
    pure_comps = [pure_component] + (['VA'] if 'VA' in comps else [])
    calc_res = calculate(dbf, pure_comps, phases, T=calc_T_cond, P=conditions[v.P])
    # Phase indices of the phase with minimum gibbs energy (over grid of T, P)
    min_energy_phase_idx = calc_res.GM.argmin(dim='points').values.squeeze()
    # transition temperature is when the next phase does not match the last phase
    # this is the lower bracket temperature (upper bracket: add one to non-zero indices)
    idx_transition_temperatures = np.nonzero(min_energy_phase_idx[:-1] != min_energy_phase_idx[1:])[0]
    transition_temperatures = np.atleast_1d(calc_res.T[idx_transition_temperatures].values.squeeze()).tolist()
    phase_sets = [] # sets of phases for each transition
    for tt_idx in idx_transition_temperatures:
        TL_phase = str(calc_res.Phase.isel(points=min_energy_phase_idx[tt_idx], T=tt_idx).values.squeeze())
        TH_phase = str(calc_res.Phase.isel(points=min_energy_phase_idx[tt_idx+1], T=tt_idx+1).values.squeeze())
        phase_sets.append({TL_phase, TH_phase})

    conds = deepcopy(conditions)
    # set the composition to near the pure element of interest
    if pure_component == dof_comp:
        conds[v.X(dof_comp)] = np.linspace(1.0-X_distance, 1.0, num_X)
    else:
        conds[v.X(dof_comp)] = np.linspace(0.0, 0.0+X_distance, num_X)
    nodes_found = []
    for temperature, current_phase_set in zip(transition_temperatures, phase_sets):
        # we don't know whether the two phase region is pointed up or down, need to search both
        trial_Ts = [
            (temperature - 2, Direction.NEGATIVE),
            (temperature + 2, Direction.POSITIVE),
        ]
        for trial_T, trial_direction in trial_Ts:
            conds[v.T] = trial_T
            hull = convex_hull(dbf, comps, phases, conds, **hull_kwargs)

            out_phases, compositions, site_fracs = hull[1], hull[3], hull[4]
            grid_shape = out_phases.shape[:-1]
            num_phases = out_phases.shape[-1]
            it = np.nditer(np.empty(grid_shape), flags=['multi_index'])  # empty grid for indexing
            while not it.finished:
                idx = it.multi_index
                trial_compsets = []
                for i in np.arange(num_phases):
                    compset = BinaryCompSet(str(out_phases[idx][i]), temperature, dof_comp, compositions[idx][i, indep_comp_idx], site_fracs[idx][i, :])
                    trial_compsets.append(compset)
                trial_phases = [c.phase_name for c in trial_compsets]
                trial_phases_set = set(trial_phases)
                trial_compositions = [c.composition for c in trial_compsets]
                # Convex hull always gives back pairs of compsets, even for true single phase regions.
                # We need to filter out regions where the phases aren't the same, those aren't true two phase regions.
                # This might break in a miscibility gap.
                # Condition 1: Number of phases must be 2
                if len(trial_phases_set) != 2:
                    it.iternext()
                    continue
                # Condition 2: Must share the phases around the transition
                if len(current_phase_set.intersection(trial_phases_set)) != 2:
                    it.iternext()
                    continue
                # If we made it here, we found a potential match!
                sp = StartPoint(trial_T - trial_direction*dT, trial_direction, trial_compsets)
                nodes_found.append(sp)
                break
    return nodes_found


