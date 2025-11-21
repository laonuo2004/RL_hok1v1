import math
import numpy as np

CUSTOM_FEATURE_SIZE = 20
_MAP_HALF_SIZE = 15000.0
_MAX_DISTANCE = 25000.0
_MAX_ATTACK_RANGE = 10000.0
_MAX_PHY_ATK = 1000.0
_MAX_MOVE_SPEED = 8000.0
_MAX_KILL_INCOME = 200.0
_MAX_HERO_LEVEL = 15.0
_MAX_FRAME_GUESS = 20000.0


def encode_monster_feature(observation, hero_camp, player_id):
    """
    Build the 20-dim custom feature vector that describes the neutral monster.
    The returned vector overwrites the reserved demo_custom_feature slot.
    """
    features = np.zeros(CUSTOM_FEATURE_SIZE, dtype=np.float32)
    if not observation:
        return features

    frame_state = observation.get("frame_state")
    if not frame_state:
        return features

    hero_state = _select_hero_state(frame_state.get("hero_states") or [], player_id, hero_camp)
    if hero_state is None:
        return features

    monster_state = _select_monster_state(frame_state.get("npc_states") or [])
    if monster_state is None:
        return features

    features[0] = 1.0  # monster exists in current observation
    monster_hp = float(monster_state.get("hp", 0.0))
    monster_max_hp = float(monster_state.get("max_hp", 1.0))
    features[1] = 1.0 if monster_hp > 0 else 0.0
    features[2] = _ratio(monster_hp, monster_max_hp)

    hero_actor_state = hero_state.get("actor_state", {})
    hero_pos = _extract_location(hero_state)
    monster_pos = _extract_location(monster_state)

    dist_self = None
    if hero_pos and monster_pos:
        raw_dx = float(monster_pos.get("x", 0.0) - hero_pos.get("x", 0.0))
        raw_dz = float(monster_pos.get("z", 0.0) - hero_pos.get("z", 0.0))
        dist_self = math.hypot(raw_dx, raw_dz)
        features[3] = _normalize_positive(dist_self, _MAX_DISTANCE)
        features[4] = _normalize_signed(raw_dx, _MAP_HALF_SIZE)
        features[5] = _normalize_signed(raw_dz, _MAP_HALF_SIZE)
        features[10] = _normalize_signed(monster_pos.get("x", 0.0), _MAP_HALF_SIZE)
        features[11] = _normalize_signed(monster_pos.get("z", 0.0), _MAP_HALF_SIZE)

    hero_runtime_id = hero_actor_state.get("runtime_id")
    features[6] = 1.0 if hero_runtime_id is not None and monster_state.get("attack_target") == hero_runtime_id else 0.0

    features[7] = _normalize_positive(monster_state.get("attack_range", 0.0), _MAX_ATTACK_RANGE)
    monster_values = monster_state.get("values") or {}
    features[8] = _normalize_positive(monster_values.get("phy_atk", 0.0), _MAX_PHY_ATK)
    features[9] = _normalize_positive(monster_values.get("mov_spd", 0.0), _MAX_MOVE_SPEED)

    features[12] = _ratio(hero_actor_state.get("hp", 0.0), hero_actor_state.get("max_hp", 1.0))
    features[13] = _normalize_positive(hero_actor_state.get("attack_range", 0.0), _MAX_ATTACK_RANGE)

    camp_visible = monster_state.get("camp_visible") or []
    camp_idx = _camp_to_index(hero_actor_state.get("camp", hero_camp))
    if isinstance(camp_visible, (list, tuple)) and 0 <= camp_idx < len(camp_visible) and camp_visible[camp_idx]:
        features[14] = 1.0
    else:
        features[14] = 0.0

    enemy_state = _select_enemy_hero(frame_state.get("hero_states") or [], hero_state)
    if enemy_state and monster_pos:
        enemy_pos = _extract_location(enemy_state)
        if enemy_pos:
            enemy_dx = float(monster_pos.get("x", 0.0) - enemy_pos.get("x", 0.0))
            enemy_dz = float(monster_pos.get("z", 0.0) - enemy_pos.get("z", 0.0))
            enemy_dist = math.hypot(enemy_dx, enemy_dz)
            features[15] = _normalize_positive(enemy_dist, _MAX_DISTANCE)
            if dist_self is not None and enemy_dist is not None and dist_self > 0:
                features[16] = 1.0 if enemy_dist < dist_self else 0.0

    features[17] = _normalize_positive(monster_state.get("kill_income", 0.0), _MAX_KILL_INCOME)
    features[18] = _normalize_positive(hero_state.get("level", 1.0), _MAX_HERO_LEVEL)
    features[19] = _normalize_positive(frame_state.get("frameNo", 0.0), _MAX_FRAME_GUESS)

    return features


def _select_hero_state(hero_states, player_id, hero_camp):
    for hero in hero_states:
        if hero.get("player_id") == player_id:
            return hero
    for hero in hero_states:
        actor_state = hero.get("actor_state", {})
        if actor_state.get("camp") == hero_camp:
            return hero
    return hero_states[0] if hero_states else None


def _select_enemy_hero(hero_states, self_hero):
    for hero in hero_states:
        if hero is self_hero:
            continue
        if hero.get("player_id") != self_hero.get("player_id"):
            return hero
        actor_state = hero.get("actor_state", {})
        self_actor_state = self_hero.get("actor_state", {})
        if actor_state.get("camp") != self_actor_state.get("camp"):
            return hero
    return None


def _select_monster_state(npc_states):
    for npc in npc_states:
        if npc.get("camp") == "PLAYERCAMP_MID" and npc.get("actor_type") == "ACTOR_MONSTER":
            sub_type = npc.get("sub_type")
            if sub_type in ("ACTOR_SUB_NONE", None):
                return npc
    return None


def _extract_location(entity):
    if not entity:
        return None
    if "actor_state" in entity and entity["actor_state"].get("location"):
        return entity["actor_state"].get("location")
    return entity.get("location")


def _normalize_signed(value, scale):
    if scale <= 0:
        return 0.0
    return max(-1.0, min(1.0, float(value) / float(scale)))


def _normalize_positive(value, scale):
    if scale <= 0:
        return 0.0
    return max(0.0, min(1.0, float(value) / float(scale)))


def _ratio(value, base):
    if base is None or base == 0:
        return 0.0
    return max(0.0, min(1.0, float(value) / float(base)))


def _camp_to_index(camp_value):
    if isinstance(camp_value, str):
        if camp_value.endswith("2"):
            return 1
        return 0
    if isinstance(camp_value, int):
        return camp_value
    return 0

