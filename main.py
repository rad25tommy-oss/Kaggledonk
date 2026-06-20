"""PTCG ABC submission entrypoint.

The simulator calls ``agent(obs_dict)``. On the very first call ``obs_dict`` has
``select == None`` and the agent must return the 60-card deck list. Later calls
must return a list of selected option indices.

This submission is intentionally self-contained: the BOSS-derived policy core
and the Donkrow policy overlay are embedded in this file so Kaggle/CABT output
only needs ``main.py`` plus ``deck.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Protocol, Sequence

# Inline BOSS policy core. Keep this submission module-free at runtime.

LEGAL_ACTION_KEYS = (
    "legal_actions",
    "legalActions",
    "legal_options",
    "legalOptions",
    "actions",
    "options",
)

OPPONENT_HIDDEN_ZONE_KEYS = ("hand", "deck", "prizes", "prize_cards", "prizeCards")
SELF_HIDDEN_ZONE_KEYS = ("deck", "prizes", "prize_cards", "prizeCards")
OPPONENT_KEYS = ("opponent", "opp", "enemy", "p2", "player2")
SELF_KEYS = ("self", "me", "player", "p1", "player1")

SUBMISSION_DEFAULT_PROFILE = "donkrow"

# BEGIN DONKROW_POLICY_JSON
POLICY_CONFIG_JSON = r'''{
  "schemaVersion": 2,
  "profile": "donkrow",
  "source": "BOSS Donkrow CPU policy overlay for PTCG ABC",
  "purpose": "Shared aliases, score weights, and behavior rules synced into the Kaggle submission package.",
  "aliases": {
    "team_rocket": [
      "team rocket",
      "rocket's",
      "ロケット団"
    ],
    "transceiver": [
      "team rocket's transceiver",
      "rocket transceiver",
      "transceiver",
      "ロケット団のレシーバー",
      "レシーバー"
    ],
    "proton": [
      "team rocket's proton",
      "proton",
      "lance",
      "ロケット団のランス",
      "ランス"
    ],
    "petrel": [
      "team rocket's petrel",
      "petrel",
      "lambda",
      "ロケット団のラムダ",
      "ラムダ"
    ],
    "ariana": [
      "team rocket's ariana",
      "ariana",
      "athena",
      "ロケット団のアテナ",
      "アテナ"
    ],
    "archer": [
      "team rocket's archer",
      "archer",
      "apollo",
      "ロケット団のアポロ",
      "アポロ"
    ],
    "giovanni": [
      "team rocket's giovanni",
      "giovanni",
      "sakaki",
      "ロケット団のサカキ",
      "サカキ"
    ],
    "factory": [
      "team rocket's factory",
      "rocket factory",
      "factory",
      "ロケット団のファクトリー",
      "ファクトリー"
    ],
    "roto_stick": [
      "roto-stick",
      "roto stick",
      "rotostick",
      "ロトりぼう"
    ],
    "pokegear": [
      "pokegear",
      "pokegear 3.0",
      "poke gear",
      "ポケギア"
    ],
    "poke_pad": [
      "poke pad",
      "pokepad",
      "ポケパッド"
    ],
    "night_stretcher": [
      "night stretcher",
      "夜のタンカ"
    ],
    "miracle_headset": [
      "miracle headset",
      "ミラクルインカム"
    ],
    "murkrow": [
      "team rocket's murkrow",
      "rocket murkrow",
      "murkrow",
      "ロケット団のヤミカラス",
      "ヤミカラス"
    ],
    "honchkrow": [
      "team rocket's honchkrow",
      "rocket honchkrow",
      "honchkrow",
      "donkrow",
      "ロケット団のドンカラス",
      "ドンカラス"
    ],
    "porygon_basic": [
      "team rocket's porygon",
      "rocket porygon",
      "porygon",
      "ロケット団のポリゴン",
      "ポリゴン"
    ],
    "porygon_attacker": [
      "team rocket's porygon2",
      "team rocket's porygon-z",
      "rocket porygon2",
      "rocket porygon-z",
      "porygon2",
      "porygon-z",
      "porygon z",
      "ロケット団のポリゴン2",
      "ロケット団のポリゴンZ",
      "ポリゴン2",
      "ポリゴンZ"
    ],
    "porygon_z": [
      "team rocket's porygon-z",
      "porygon-z",
      "porygon z",
      "ロケット団のポリゴンZ",
      "ポリゴンZ"
    ],
    "rocket_pokemon": [
      "team rocket's murkrow",
      "team rocket's honchkrow",
      "team rocket's porygon",
      "team rocket's porygon2",
      "team rocket's porygon-z",
      "ロケット団のヤミカラス",
      "ロケット団のドンカラス",
      "ロケット団のポリゴン",
      "ロケット団のポリゴン2",
      "ロケット団のポリゴンZ"
    ],
    "team_rocket_energy": [
      "team rocket's energy",
      "rocket energy",
      "ロケット団のエネルギー"
    ],
    "ignition_energy": [
      "ignition energy",
      "イグニッションエネルギー"
    ],
    "rocket_feather": [
      "rocket feather",
      "ロケットフェザー"
    ],
    "taunt": [
      "taunt",
      "ちょうはつ"
    ],
    "ko_or_prize": [
      "ko",
      "knock out",
      "knock",
      "prize",
      "take prize",
      "きぜつ",
      "サイド"
    ],
    "active": [
      "active",
      "battle active",
      "バトル場",
      "前"
    ],
    "bench": [
      "bench",
      "bench-",
      "ベンチ",
      "後ろ"
    ],
    "immediate_attack": [
      "immediate",
      "this turn",
      "attack now",
      "can attack",
      "ready attack",
      "この番",
      "すぐ",
      "ワザを使える"
    ],
    "future_plan": [
      "future",
      "next turn",
      "later",
      "backup",
      "次",
      "次の番",
      "後続"
    ],
    "proactive_plan": [
      "prepare",
      "setup",
      "plan",
      "proactive",
      "準備",
      "育成"
    ]
  },
  "rules": {
    "avoidNonKoRocketFeatherIntoHealing": {
      "enabled": true,
      "zeroFuelScore": -22000,
      "koBaseScore": 16500,
      "koPerRequiredSupporterScore": 700,
      "lowDeckKoBonus": 2400,
      "baseNonKoScore": 900,
      "perSupporterNonKoScore": 400,
      "missingKoSupporterPenalty": 1800,
      "oneSupporterPenalty": 4800,
      "hp120Penalty": 2800,
      "hp180Penalty": 2000,
      "lowDeckThreshold": 10,
      "lowDeckPenalty": 5600,
      "setupOrSwitchPenalty": 1600,
      "bossReadyAttackPenalty": 70000,
      "bossRankPenalty": 90000
    },
    "compressBeforeAriana": {
      "enabled": true,
      "highHandMin": 7,
      "lowDrawMax": 2,
      "lowDrawPenalty": 12000,
      "compressionHandMin": 6,
      "compressionPenalty": 8500,
      "maxLowValueScore": 1800
    },
    "preferProtonWhenSetupIncomplete": {
      "enabled": true,
      "earlyTurnMax": 4,
      "mainEarlyScore": 14400,
      "mainSetupScore": 11600,
      "transceiverSetupBonus": 3200,
      "pokegearSetupBonus": 2600,
      "settledSearchPenalty": 50000,
      "settledMainScore": -50000,
      "settledRecoveryScore": -50000,
      "arianaOverProtonPenalty": 2400,
      "arianaPokegearCompressionPenalty": 1800,
      "arianaSearchCompressionPenalty": 1400
    },
    "avoidIgnitionWaste": {
      "enabled": true,
      "forbidIgnitionOnMurkrowOrHonchkrow": false,
      "forbidIgnitionOnBasicMurkrow": true,
      "allowImmediateIgnitionOnHonchkrow": true,
      "requireActiveOrSwitchForPorygon2": true,
      "honchkrowActiveAttackScore": 15800,
      "honchkrowSwitchableAttackScore": 10200,
      "porygon2ActiveScore": 11800,
      "porygon2SwitchableScore": 8600,
      "benchWithoutSwitchPenalty": -15000,
      "duplicateIgnitionPenalty": -11000,
      "forbiddenTargetScore": -16000
    },
    "preferHonchkrowPromotion": {
      "enabled": true,
      "honchkrowScore": 26000,
      "porygon2Score": 22500,
      "murkrowWithAttackerPenalty": -18000,
      "porygonWithAttackerPenalty": -12000,
      "fallbackScore": 1200
    },
    "avoidMurkrowEnergyWhenAttackerReady": {
      "enabled": true,
      "betterAttackerPenalty": -17000,
      "existingHonchkrowNeedsEnergyPenalty": -15000,
      "evolveableMurkrowScore": 7600,
      "earlyMurkrowScore": 5200,
      "activeAttackReadyMurkrowScore": 1800,
      "duplicateRocketEnergyPenalty": -20000,
      "nextAttackerPreparationBonus": 2400
    },
    "avoidLowValueSwitching": {
      "enabled": true,
      "noReadyBenchAttackerPenalty": -18000,
      "activeAttackerPenalty": -22000,
      "readyBenchAttackerScore": 7200
    },
    "sakakiRequiresKo": {
      "enabled": true,
      "nonKoScore": -9000,
      "searchPenalty": -3200,
      "koBaseScore": 95000,
      "koPerPrizeScore": 75000,
      "minPrizeScore": 2
    },
    "rocketApolloReset": {
      "enabled": true,
      "unavailableScore": -9500,
      "avoidScore": -8500,
      "searchPenalty": 2200,
      "baseScore": 7500,
      "shortHandScore": 8000,
      "mediumHandScore": 4000,
      "largeHandScore": 1200,
      "recoveryBaseScore": 18000,
      "murkrowLineBonus": 4500,
      "noHandEnergyBonus": 8500,
      "needsEnergyBonus": 10000,
      "needsFuelBonus": 6500,
      "perEvolutionScore": 2400,
      "opponentHandPressureScore": 500,
      "lowDeckResetThreshold": 6,
      "fuelOnlyScore": 900
    },
    "donkrowTurnPlan": {
      "enabled": true,
      "attackBaseScore": 24000,
      "koPrizeScore": 12500,
      "gameEndScore": 85000,
      "damageScore": 18,
      "energyAttachScore": 26000,
      "switchScore": 30000,
      "benchPlanPenalty": 4200,
      "honchkrowPlanBonus": 12000,
      "murkrowAttackPlanPenalty": 28000,
      "murkrowNonKoPenalty": 18000,
      "porygon2NonKoPenalty": 8000
    },
    "porygon2RCommandFallback": {
      "enabled": true,
      "damagePerTrashRocketSupporter": 20,
      "zeroDamageScore": -12000,
      "koScore": 14800,
      "koPerSupporterScore": 250,
      "nonKoBaseScore": 6200,
      "nonKoPerSupporterScore": 520,
      "lowDeckThreshold": 10,
      "lowDeckPenalty": 2800
    },
    "preferArianaEnergyDig": {
      "enabled": true,
      "noHandEnergyBonus": 9200,
      "attackStarvedBonus": 10500,
      "transceiverArianaBonus": 8800,
      "pokegearArianaBonus": 5200,
      "petrelArianaBonus": 4600,
      "factoryBeforeArianaPenalty": 6000
    },
    "pokePadEvolutionAttack": {
      "enabled": true,
      "honchkrowWithEnergyInHandScore": 12000,
      "porygon2WithIgnitionInHandScore": 11800,
      "evolutionBackupScore": 5600,
      "activeEvolutionBonus": 2800
    },
    "attackContinuity": {
      "enabled": true,
      "activeHonchkrowEvolutionScore": 42000,
      "benchHonchkrowEvolutionScore": 28000,
      "nextAttackerRocketEnergyBonus": 21000,
      "activeReadyBasicBenchScore": 32000,
      "activeReadyPokePadScore": 30000
    },
    "auxiliaryTauntPlan": {
      "enabled": true,
      "readyOpponentScore": 3600,
      "notReadyPenalty": -12000,
      "damageBonus": 18,
      "setupPenalty": 1400
    }
  },
  "weights": {
    "identity": {
      "transceiver": 180,
      "proton": -600,
      "petrel": 140,
      "ariana": 220,
      "archer": 30,
      "giovanni": 100,
      "factory": 140,
      "rotoStick": 70,
      "pokegear": 65,
      "pokePad": 75,
      "nightStretcher": 55,
      "miracleHeadset": 95
    },
    "board": {
      "benchMurkrow": 240,
      "benchMurkrowWhenNoAttack": 110,
      "benchPorygon": 130,
      "benchPorygonWhenNoAttack": 80,
      "benchRocketPokemon": 70,
      "evolveHonchkrow": 300,
      "evolveHonchkrowWithEnergy": 100,
      "evolvePorygonAttacker": 180,
      "evolvePorygonZ": 140
    },
    "search": {
      "searchBeforeFallback": 90,
      "transceiverBeforeBench": 140,
      "transceiverBeforeEnergy": 80,
      "transceiverWhenNoAttack": 160,
      "protonFindsBasics": -220,
      "petrelFindsEnergyRoute": 130
    },
    "supporter": {
      "openSupporterRight": 70,
      "usedSupporterPenalty": -280,
      "preserveAthenaPrizeRaceAhead": 380
    },
    "energy": {
      "openEnergyRight": 60,
      "usedEnergyPenalty": -450,
      "prepareWhenNoAttack": 220,
      "prepareBenchWhenNoAttack": 170,
      "immediateAttack": 360,
      "immediateKo": 260,
      "futurePlan": 160,
      "proactivePlan": 70,
      "honchkrowTarget": 240,
      "porygonAttackerTarget": 210,
      "murkrowTarget": 65,
      "murkrowNonImmediatePenalty": -120,
      "porygonBasicTarget": 55,
      "teamRocketEnergyOnPorygonPenalty": -260,
      "ignitionOnPorygonAttacker": 420,
      "porygonNonIgnitionPenalty": -720,
      "benchRocketEnergyWhileAttacking": 260,
      "honchkrowIgnitionRedundantPenalty": -900
    },
    "attack": {
      "rocketFeather": 420,
      "rocketFeatherKo": 260,
      "koOrPrize": 240,
      "tauntWithoutKoPenalty": -520,
      "nonKoBeforeSetupPenalty": -140,
      "rocketFeatherAthenaDiscardPrizeRacePenalty": -430,
      "rocketFeatherCostLance": 160,
      "rocketFeatherCostGiovanni": 100,
      "rocketFeatherCostArcher": 50,
      "rocketFeatherCostPetrel": 0,
      "rocketFeatherCostAthena": -260,
      "rocketFeatherCostAthenaNoNextSupporterPenalty": -1200
    },
    "turnRights": {
      "factoryStadiumRight": 120,
      "factoryWithRocketSupporter": 160
    },
    "cat": {
      "earlyHonchkrow": 220,
      "earlyMurkrow": 150,
      "earlySearchSupport": 130,
      "rocketFeatherAvailable": 180,
      "delayKoForValuePenalty": -160,
      "lowDeckKo": 280,
      "lowDeckDrawPenalty": -300
    },
    "penalty": {
      "passWithSearch": -260,
      "passWithEnergy": -320,
      "passWithBench": -220,
      "passWithAttack": -200
    }
  }
}
'''
# END DONKROW_POLICY_JSON
POLICY_CONFIG: Mapping[str, Any] = json.loads(POLICY_CONFIG_JSON)

OPTION_TYPE_NAMES = {
    1: "Yes",
    2: "No",
    3: "Card",
    4: "Number",
    7: "Play",
    8: "Attach",
    9: "Evolve",
    10: "Ability",
    12: "Retreat",
    13: "Attack",
    14: "End",
}

AREA_TYPE_NAMES = {
    1: "deck",
    2: "hand",
    3: "discard",
    4: "active",
    5: "bench",
    6: "prize",
    7: "stadium",
    12: "looking",
}

SELECT_CONTEXT_NAMES = {
    0: "main",
    1: "setup active pokemon",
    2: "setup bench pokemon",
    7: "to hand search",
    8: "rocket feather discard cost",
    41: "deck registration",
}

CARD_ID_NAMES = {
    463: "Team Rocket's Murkrow",
    891: "Team Rocket's Honchkrow",
    474: "Team Rocket's Porygon2",
    473: "Team Rocket's Porygon",
    1134: "Team Rocket's Transceiver",
    1077: "Roto-Stick",
    1152: "Poke Pad",
    1097: "Night Stretcher",
    1109: "Miracle Headset",
    1220: "Team Rocket's Proton",
    1219: "Team Rocket's Petrel",
    1218: "Team Rocket's Giovanni",
    1217: "Team Rocket's Archer",
    1216: "Team Rocket's Ariana",
    1257: "Team Rocket's Factory",
    15: "Team Rocket's Energy",
    17: "Ignition Energy",
    1122: "Pokegear 3.0",
}

POKEMON_CARD_IDS = {463, 891, 474, 473}
ENERGY_CARD_IDS = {15, 17}
STADIUM_CARD_IDS = {1257}
SUPPORTER_CARD_IDS = {1220, 1219, 1218, 1217, 1216}
ITEM_CARD_IDS = {1134, 1077, 1152, 1097, 1109, 1122}

ATTACK_ID_NAMES = {
    652: "Murkrow Tempt",
    653: "Murkrow secondary attack",
    669: "Porygon attack",
    670: "Porygon2 R Command",
    1285: "Rocket Feather",
    1286: "Taunt",
}


def policy_value(path: str, fallback: Any = None) -> Any:
    current: Any = POLICY_CONFIG
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return fallback
        current = current[part]
    return current


def policy_weight(path: str, fallback: float = 0.0) -> float:
    value = policy_value(path, fallback)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return float(value)


def policy_terms(alias_key: str, fallback: Sequence[str]) -> tuple[str, ...]:
    value = policy_value(f"aliases.{alias_key}", None)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return tuple(term.lower() for term in fallback)

    terms = tuple(str(term).lower() for term in value if term)
    return terms or tuple(term.lower() for term in fallback)


@dataclass(frozen=True)
class Candidate:
    index: int
    raw: Any
    text: str
    compact: str
    kind: str


@dataclass(frozen=True)
class PolicyContext:
    observation: Any
    profile_name: str
    observation_text: str
    legal_action_texts: tuple[str, ...]
    has_attack_action: bool
    has_non_taunt_attack: bool
    has_ko_attack: bool
    has_rocket_feather_action: bool
    has_energy_action: bool
    has_bench_action: bool
    has_search_action: bool
    energy_right_used: bool
    supporter_right_used: bool
    stadium_right_used: bool
    self_hand_count: int | None
    self_deck_count: int | None
    self_prize_count: int | None
    opponent_prize_count: int | None
    turn_number: int | None


class DeckPolicy(Protocol):
    name: str

    def matches(self, context: PolicyContext) -> bool:
        ...

    def score(self, candidate: Candidate, context: PolicyContext) -> float:
        ...


def is_prize_race_ahead(context: PolicyContext) -> bool:
    return (
        context.self_prize_count is not None
        and context.opponent_prize_count is not None
        and context.self_prize_count < context.opponent_prize_count
    )


class GlobalPokemonPolicy:
    """Deck-agnostic priorities that should survive across archetypes."""

    def score(self, candidate: Candidate, context: PolicyContext) -> float:
        text = candidate.compact
        score = {
            "attack": 900.0,
            "ability": 760.0,
            "evolve": 740.0,
            "attach_energy": 710.0,
            "trainer": 640.0,
            "bench": 520.0,
            "stadium": 500.0,
            "switch": 340.0,
            "other": 100.0,
            "pass": -350.0,
        }[candidate.kind]

        if candidate.kind == "attack":
            score += damage_bonus(text)
            if contains_any(text, ("ko", "knock", "prize", "take prize", "きぜつ", "サイド", "倒")):
                score += 520.0
            if contains_any(text, ("active", "バトル場", "前")):
                score += 30.0
        elif candidate.kind == "trainer":
            if contains_any(text, ("search", "山札", "サーチ", "手札に加", "ball", "ボール")):
                score += 130.0
            if contains_any(text, ("draw", "ドロー", "引")):
                score += 90.0
            if contains_any(text, ("boss", "gust", "switch opponent", "呼び出", "入れ替え")):
                score += 140.0
        elif candidate.kind == "attach_energy":
            if contains_any(text, ("active", "attacker", "バトル場", "アタッカー", "ワザ")):
                score += 80.0
        elif candidate.kind == "evolve":
            if contains_any(text, ("stage2", "2進化", "ex")):
                score += 120.0
        elif candidate.kind == "bench":
            if contains_any(text, ("basic", "たね", "ベンチ")):
                score += 70.0
        elif candidate.kind == "pass":
            if has_non_pass_actions(context.observation):
                score -= 500.0

        score -= candidate.index * 0.001
        return score


class DonkrowPolicy:
    """Team Rocket's Honchkrow deck priorities extracted from BOSS CPU work."""

    name = "donkrow"

    deck_terms = (
        "team rocket",
        "rocket's",
        "ロケット団",
        "honchkrow",
        "donkrow",
        "murkrow",
        "porygon",
        "roto-stick",
        "roto stick",
        "roto",
        "poké pad",
        "poke pad",
        "ポケパッド",
        "ロトりぼう",
        "ドンカラス",
        "ヤミカラス",
        "ポリゴン",
    )

    def matches(self, context: PolicyContext) -> bool:
        if context.profile_name == self.name:
            return True
        return contains_any(context.observation_text, self.deck_terms)

    def score(self, candidate: Candidate, context: PolicyContext) -> float:
        return (
            self._score_card_identity(candidate, context)
            + self._score_board_development(candidate, context)
            + self._score_search_ladder(candidate, context)
            + self._score_supporter_plan(candidate, context)
            + self._score_energy_plan(candidate, context)
            + self._score_attack_plan(candidate, context)
            + self._score_turn_rights(candidate, context)
            + self._score_cat_tendency_adjustments(candidate, context)
            + self._score_policy_config_overlay(candidate, context)
            + self._score_low_value_penalties(candidate, context)
        )

    def _score_card_identity(self, candidate: Candidate, context: PolicyContext) -> float:
        text = candidate.compact
        score = 0.0

        if contains_any(text, ("team rocket's transceiver", "rocket transceiver", "ロケット団のレシーバー")):
            score += 900.0
            if not context.supporter_right_used:
                score += 220.0
        if is_proton(text):
            score += 760.0
        if is_petrel(text):
            score += 720.0
        if is_ariana(text):
            score += 560.0
        if is_archer(text):
            score += 450.0
        if is_giovanni(text):
            score += 520.0

        if is_factory(text):
            score += 680.0
            if not context.stadium_right_used:
                score += 160.0
        if is_roto_stick(text):
            score += 470.0
        if is_pokegear(text):
            score += 390.0
        if is_poke_pad(text):
            score += 440.0
        if is_night_stretcher(text):
            score += 380.0
        if is_miracle_headset(text):
            score += 540.0

        return score

    def _score_board_development(self, candidate: Candidate, context: PolicyContext) -> float:
        text = candidate.compact
        score = 0.0

        if candidate.kind == "bench":
            if is_murkrow(text):
                score += 900.0
                if not context.has_non_taunt_attack:
                    score += 260.0
            if is_porygon_basic(text):
                score += 620.0
                if not context.has_non_taunt_attack:
                    score += 220.0
            if is_rocket_pokemon(text):
                score += 220.0
            if contains_any(text, ("meowth", "mimikyu", "articuno", "ミミッキュ", "フリーザー")):
                score += 180.0

        if candidate.kind == "evolve":
            if is_honchkrow(text):
                score += 1_150.0
                if context.has_energy_action and not context.has_non_taunt_attack:
                    score += 180.0
            if is_porygon_attacker(text):
                score += 780.0
            if is_porygon_z(text):
                score += 660.0

        return score

    def _score_search_ladder(self, candidate: Candidate, context: PolicyContext) -> float:
        text = candidate.compact
        score = 0.0

        # BOSS CAT tended to improve when the CPU used search before giving up,
        # especially Transceiver -> Proton/Petrel and Petrel -> missing trainer.
        if candidate.kind == "trainer":
            if contains_any(text, ("search", "deck", "山札", "手札に加", "choose", "select")):
                score += 180.0
            if is_transceiver(text):
                if context.has_bench_action:
                    score += 240.0
                if context.has_energy_action:
                    score += 120.0
                if not context.has_non_taunt_attack:
                    score += 240.0
            if is_proton(text):
                score += 280.0
                if context.has_bench_action:
                    score += 260.0
            if is_petrel(text):
                score += 240.0
                if context.has_energy_action:
                    score += 160.0
                if not context.has_non_taunt_attack:
                    score += 220.0
            if is_pokegear(text) or is_poke_pad(text) or is_roto_stick(text):
                score += 180.0
                if not context.has_non_taunt_attack:
                    score += 160.0
            if is_night_stretcher(text) or is_miracle_headset(text):
                if "trash" in text or "トラッシュ" in text:
                    score += 170.0

        return score

    def _score_supporter_plan(self, candidate: Candidate, context: PolicyContext) -> float:
        text = candidate.compact
        score = 0.0
        has_ariana_option = any(is_ariana(action_text) for action_text in context.legal_action_texts)
        has_miracle_headset_option = any(is_miracle_headset(action_text) for action_text in context.legal_action_texts)

        if not candidate.kind == "trainer":
            return score

        if context.supporter_right_used and is_rocket_supporter(text):
            return score - 650.0

        if is_proton(text):
            # Proton is the best early board-construction supporter.
            score -= 1_600.0
            if not context.has_non_taunt_attack:
                score += 300.0
            if context.has_bench_action:
                score += 260.0
            if contains_any(text, ("murkrow", "porygon", "basic", "ヤミカラス", "ポリゴン", "たね")):
                score += 260.0

        if is_petrel(text):
            # Petrel is the bridge to the exact trainer: Switch, Factory,
            # Transceiver follow-up, Pad/Gear, recovery, or energy access.
            if contains_any(text, ("trainer", "item", "stadium", "supporter", "トレーナーズ", "グッズ", "スタジアム")):
                score += 260.0
            if contains_any(text, ("switch", "factory", "energy", "pad", "gear", "roto", "いれかえ")):
                score += 260.0

        if is_ariana(text):
            # Ariana is draw/fuel. It matters more when the attack is starved.
            if context.self_hand_count is not None:
                if context.self_hand_count <= 4:
                    score += 1_100.0
                elif context.self_hand_count <= 6:
                    score += 650.0
            if not context.has_non_taunt_attack:
                score += 780.0
            if context.has_rocket_feather_action:
                score += 700.0
            if context.has_energy_action:
                score += 360.0
            if contains_any(context.observation_text, ("team rocket's factory", "rocket factory", "ロケット団のファクトリー")):
                score += 520.0
            if is_prize_race_ahead(context) and context.has_rocket_feather_action and not context.has_ko_attack:
                score += policy_weight("weights.supporter.preserveAthenaPrizeRaceAhead")
            if not context.supporter_right_used and has_miracle_headset_option:
                score += 760.0
                
        if is_miracle_headset(text):
            if context.has_ko_attack:
                score += 420.0
            else:
                score -= 900.0
            if not context.supporter_right_used and has_ariana_option:
                score -= 520.0
    

        if is_archer(text):
            # Archer is strong after losing a Pokemon, but weaker as a blind
            # default before board construction.
            if contains_any(context.observation_text, ("lost", "ko last turn", "lostmon", "きぜつ", "倒された")):
                score += 260.0
            elif not context.has_non_taunt_attack:
                score -= 140.0

        if is_giovanni(text):
            if context.has_ko_attack:
                score += 280.0
            if contains_any(text, ("switch", "gust", "bench", "呼び出", "入れ替え")):
                score += 240.0

        return score

    def _score_energy_plan(self, candidate: Candidate, context: PolicyContext) -> float:
        text = candidate.compact
        score = 0.0

        if candidate.kind != "attach_energy":
            return score

        if context.energy_right_used:
            return score - 1_000.0

        # Attack comes after setup in BOSS. If an energy action is available and
        # the best attack needs it, spend the right before attacking. If no real
        # attack exists, prepare the next attacker on the bench instead.
        if not context.has_non_taunt_attack:
            score += 420.0
            if mentions_bench(text):
                score += 360.0
            if is_honchkrow(text):
                score += 720.0
            if is_murkrow(text):
                score += 460.0
            if is_porygon_attacker(text):
                score += 620.0
            if is_porygon_basic(text):
                score += 220.0
        else:
            if mentions_active(text) and (is_honchkrow(text) or is_porygon_attacker(text)):
                score += 260.0
            if mentions_bench(text):
                score += 180.0

        if mentions_immediate_attack(text):
            score += 1_200.0
            if is_honchkrow(text) or is_porygon_attacker(text):
                score += 650.0
            if is_ko_or_prize(text):
                score += 700.0
        elif mentions_future_plan(text):
            score += 520.0
            if mentions_bench(text):
                score += 300.0
        elif mentions_proactive_plan(text):
            score += 260.0
            if mentions_bench(text):
                score += 220.0

        if is_honchkrow(text):
            score += 680.0 + 17_000.0 / 100.0
        if is_porygon_attacker(text):
            score += 600.0 + 15_000.0 / 100.0
        if is_murkrow(text):
            score += 240.0 + 9_000.0 / 100.0
        if is_porygon_basic(text):
            score += 8_000.0 / 100.0

        if is_team_rocket_energy(text):
            score += 260.0
            if not is_rocket_pokemon(text):
                score -= 1_200.0
            if is_porygon_basic(text) or is_porygon_attacker(text):
                score -= 1_400.0
            if is_honchkrow(text) and context.has_rocket_feather_action and not mentions_immediate_attack(text):
                score -= 1_200.0
            if (is_honchkrow(text) or is_murkrow(text)) and mentions_bench(text) and context.has_non_taunt_attack:
                score += policy_weight("weights.energy.benchRocketEnergyWhileAttacking")

        if is_ignition_energy(text):
            if is_porygon_attacker(text):
                score += 1_250.0
                if mentions_immediate_attack(text):
                    score += 620.0
            elif is_honchkrow(text):
                score += 320.0
                if context.has_rocket_feather_action and contains_any(context.observation_text, policy_terms("team_rocket_energy", ("team rocket's energy", "rocket energy"))):
                    score += policy_weight("weights.energy.honchkrowIgnitionRedundantPenalty")
            elif is_murkrow(text):
                if context.has_non_taunt_attack or not mentions_immediate_attack(text):
                    score += policy_weight("weights.energy.honchkrowIgnitionRedundantPenalty") / 2
            elif is_porygon_basic(text):
                score -= 1_600.0
            else:
                score -= 180.0
        elif is_porygon_attacker(text):
            # BOSS reserves Porygon2/Z attack turns for Ignition Energy instead
            # of slowly spending ordinary manual attachments when possible.
            score += policy_weight("weights.energy.porygonNonIgnitionPenalty")
            if not is_ko_or_prize(text) and not mentions_immediate_attack(text):
                score -= 520.0

        if is_murkrow(text) and not mentions_immediate_attack(text):
            score -= 220.0

        if contains_any(text, ("over", "too much", "excess", "過剰")):
            score -= 500.0

        return score

    def _score_attack_plan(self, candidate: Candidate, context: PolicyContext) -> float:
        text = candidate.compact
        score = 0.0

        if candidate.kind != "attack":
            return score

        if is_rocket_feather(text):
            score += 1_050.0
            if is_ko_or_prize(text):
                score += 760.0
            if is_prize_race_ahead(context) and is_ariana(text) and not is_ko_or_prize(text):
                score += policy_weight("weights.attack.rocketFeatherAthenaDiscardPrizeRacePenalty")
            score += score_rocket_feather_discard_cost(text)
            if contains_any(text, ("discard supporter", "rocket supporter", "supporter cost", "サポート", "トラッシュ")):
                score += 220.0

        if is_ko_or_prize(text):
            score += 600.0

        if is_taunt(text):
            if is_ko_or_prize(text):
                score += 120.0
            else:
                score -= 1_100.0
                if context.has_search_action or context.has_energy_action or context.has_bench_action:
                    score -= 300.0

        if is_porygon_attacker(text):
            score += 300.0
        if is_murkrow(text) and not is_ko_or_prize(text):
            score -= 180.0

        # Attack is turn-ending. Keep it behind meaningful setup unless it wins
        # a prize now.
        if not is_ko_or_prize(text) and (context.has_search_action or context.has_energy_action or context.has_bench_action):
            score -= 260.0
        if is_ko_or_prize(text):
            score += 260.0

        return score

    def _score_turn_rights(self, candidate: Candidate, context: PolicyContext) -> float:
        text = candidate.compact
        score = 0.0

        if candidate.kind in ("stadium", "ability") and is_factory(text):
            if context.stadium_right_used:
                score -= 360.0
            elif context.supporter_right_used:
                score += 980.0
            else:
                score -= 260.0
                if any(is_rocket_supporter(action) for action in context.legal_action_texts):
                    score -= 180.0

        if candidate.kind == "trainer" and is_rocket_supporter(text) and not context.supporter_right_used:
            score += 180.0
            if contains_any(context.observation_text, ("team rocket's factory", "ロケット団のファクトリー")):
                score += 220.0

        if candidate.kind == "switch":
            if contains_any(text, ("honchkrow", "porygon2", "porygon-z", "ドンカラス", "ポリゴン2", "ポリゴンz")):
                score += 340.0
            if contains_any(text, ("ドンカラス", "honchkrow")) and _donkarasu_can_ko_from_bench(context.observation):
                current, your_index, player = _current_player(context.observation)
                active_card = _top_card(_read(player, "active", []))
                if _card_id(active_card) != HONCHKROW:
                    score += 2000.0
            if context.has_ko_attack:
                score += 220.0

        if candidate.kind == "pass":
            if context.has_search_action:
                score -= 480.0
            if context.has_energy_action and not context.energy_right_used:
                score -= 560.0
            if context.has_bench_action:
                score -= 420.0
            if context.has_attack_action:
                score -= 300.0

        return score

    def _score_policy_config_overlay(self, candidate: Candidate, context: PolicyContext) -> float:
        """Apply synced CAT/BOSS policy weights from the embedded policy block."""

        text = candidate.compact
        score = 0.0

        if is_transceiver(text):
            score += policy_weight("weights.identity.transceiver")
        if is_proton(text):
            score += policy_weight("weights.identity.proton")
        if is_petrel(text):
            score += policy_weight("weights.identity.petrel")
        if is_ariana(text):
            score += policy_weight("weights.identity.ariana")
        if is_archer(text):
            score += policy_weight("weights.identity.archer")
        if is_giovanni(text):
            score += policy_weight("weights.identity.giovanni")
        if is_factory(text):
            score += policy_weight("weights.identity.factory")
        if is_roto_stick(text):
            score += policy_weight("weights.identity.rotoStick")
        if is_pokegear(text):
            score += policy_weight("weights.identity.pokegear")
        if is_poke_pad(text):
            score += policy_weight("weights.identity.pokePad")
        if is_night_stretcher(text):
            score += policy_weight("weights.identity.nightStretcher")
        if is_miracle_headset(text):
            score += policy_weight("weights.identity.miracleHeadset")

        if candidate.kind == "bench":
            if is_murkrow(text):
                score += policy_weight("weights.board.benchMurkrow")
                if not context.has_non_taunt_attack:
                    score += policy_weight("weights.board.benchMurkrowWhenNoAttack")
            if is_porygon_basic(text):
                score += policy_weight("weights.board.benchPorygon")
                if not context.has_non_taunt_attack:
                    score += policy_weight("weights.board.benchPorygonWhenNoAttack")
            if is_rocket_pokemon(text):
                score += policy_weight("weights.board.benchRocketPokemon")

        if candidate.kind == "evolve":
            if is_honchkrow(text):
                score += policy_weight("weights.board.evolveHonchkrow")
                if context.has_energy_action and not context.has_non_taunt_attack:
                    score += policy_weight("weights.board.evolveHonchkrowWithEnergy")
            if is_porygon_attacker(text):
                score += policy_weight("weights.board.evolvePorygonAttacker")
            if is_porygon_z(text):
                score += policy_weight("weights.board.evolvePorygonZ")

        if candidate.kind == "trainer":
            if is_search_or_draw_action(candidate):
                score += policy_weight("weights.search.searchBeforeFallback")
            if is_transceiver(text):
                if context.has_bench_action:
                    score += policy_weight("weights.search.transceiverBeforeBench")
                if context.has_energy_action:
                    score += policy_weight("weights.search.transceiverBeforeEnergy")
                if not context.has_non_taunt_attack:
                    score += policy_weight("weights.search.transceiverWhenNoAttack")
            if is_proton(text) and context.has_bench_action:
                score += policy_weight("weights.search.protonFindsBasics")
            if is_petrel(text) and context.has_energy_action:
                score += policy_weight("weights.search.petrelFindsEnergyRoute")
            if context.supporter_right_used and is_rocket_supporter(text):
                score += policy_weight("weights.supporter.usedSupporterPenalty")
            elif is_rocket_supporter(text):
                score += policy_weight("weights.supporter.openSupporterRight")
            if is_ariana(text) and is_prize_race_ahead(context) and context.has_rocket_feather_action and not context.has_ko_attack:
                score += policy_weight("weights.supporter.preserveAthenaPrizeRaceAhead")

        if candidate.kind == "attach_energy":
            if context.energy_right_used:
                score += policy_weight("weights.energy.usedEnergyPenalty")
            else:
                score += policy_weight("weights.energy.openEnergyRight")
            if not context.has_non_taunt_attack:
                score += policy_weight("weights.energy.prepareWhenNoAttack")
                if mentions_bench(text):
                    score += policy_weight("weights.energy.prepareBenchWhenNoAttack")
            if mentions_immediate_attack(text):
                score += policy_weight("weights.energy.immediateAttack")
                if is_ko_or_prize(text):
                    score += policy_weight("weights.energy.immediateKo")
            elif mentions_future_plan(text):
                score += policy_weight("weights.energy.futurePlan")
            elif mentions_proactive_plan(text):
                score += policy_weight("weights.energy.proactivePlan")
            if is_honchkrow(text):
                score += policy_weight("weights.energy.honchkrowTarget")
            if is_porygon_attacker(text):
                score += policy_weight("weights.energy.porygonAttackerTarget")
            if is_murkrow(text):
                score += policy_weight("weights.energy.murkrowTarget")
                if not mentions_immediate_attack(text):
                    score += policy_weight("weights.energy.murkrowNonImmediatePenalty")
            if is_porygon_basic(text):
                score += policy_weight("weights.energy.porygonBasicTarget")
            if is_team_rocket_energy(text) and is_porygon_attacker(text):
                score += policy_weight("weights.energy.teamRocketEnergyOnPorygonPenalty")
            if is_team_rocket_energy(text) and (is_honchkrow(text) or is_murkrow(text)) and mentions_bench(text) and context.has_non_taunt_attack:
                score += policy_weight("weights.energy.benchRocketEnergyWhileAttacking")
            if is_ignition_energy(text) and is_porygon_attacker(text):
                score += policy_weight("weights.energy.ignitionOnPorygonAttacker")
            if is_porygon_attacker(text) and not is_ignition_energy(text):
                score += policy_weight("weights.energy.porygonNonIgnitionPenalty")
            if is_ignition_energy(text) and (is_honchkrow(text) or is_murkrow(text)) and context.has_rocket_feather_action:
                score += policy_weight("weights.energy.honchkrowIgnitionRedundantPenalty")

        if candidate.kind == "attack":
            if is_rocket_feather(text):
                score += policy_weight("weights.attack.rocketFeather")
                if is_ko_or_prize(text):
                    score += policy_weight("weights.attack.rocketFeatherKo")
                if is_prize_race_ahead(context) and is_ariana(text) and not is_ko_or_prize(text):
                    score += policy_weight("weights.attack.rocketFeatherAthenaDiscardPrizeRacePenalty")
                score += score_rocket_feather_discard_cost(text)
            if is_ko_or_prize(text):
                score += policy_weight("weights.attack.koOrPrize")
            if is_taunt(text) and not is_ko_or_prize(text):
                score += policy_weight("weights.attack.tauntWithoutKoPenalty")
            if not is_ko_or_prize(text) and (
                context.has_search_action or context.has_energy_action or context.has_bench_action
            ):
                score += policy_weight("weights.attack.nonKoBeforeSetupPenalty")

        if candidate.kind == "stadium" and is_factory(text) and not context.stadium_right_used:
            score += policy_weight("weights.turnRights.factoryStadiumRight")
            if any(is_rocket_supporter(action) for action in context.legal_action_texts):
                score += policy_weight("weights.turnRights.factoryWithRocketSupporter")

        if candidate.kind == "pass":
            if context.has_search_action:
                score += policy_weight("weights.penalty.passWithSearch")
            if context.has_energy_action and not context.energy_right_used:
                score += policy_weight("weights.penalty.passWithEnergy")
            if context.has_bench_action:
                score += policy_weight("weights.penalty.passWithBench")
            if context.has_attack_action:
                score += policy_weight("weights.penalty.passWithAttack")

        if context.turn_number is not None and context.turn_number <= 2:
            if candidate.kind == "evolve" and is_honchkrow(text):
                score += policy_weight("weights.cat.earlyHonchkrow")
            if candidate.kind == "bench" and is_murkrow(text):
                score += policy_weight("weights.cat.earlyMurkrow")
            if candidate.kind == "trainer" and (is_proton(text) or is_transceiver(text) or is_poke_pad(text)):
                score += policy_weight("weights.cat.earlySearchSupport")

        if context.has_rocket_feather_action:
            if candidate.kind == "attack" and is_rocket_feather(text):
                score += policy_weight("weights.cat.rocketFeatherAvailable")
            if candidate.kind in ("trainer", "ability", "stadium") and context.has_ko_attack:
                score += policy_weight("weights.cat.delayKoForValuePenalty")

        if context.self_deck_count is not None and context.self_deck_count <= 6:
            if candidate.kind == "attack" and is_ko_or_prize(text):
                score += policy_weight("weights.cat.lowDeckKo")
            if candidate.kind in ("trainer", "ability", "stadium") and is_search_or_draw_action(candidate):
                score += policy_weight("weights.cat.lowDeckDrawPenalty")

        return score

    def _score_low_value_penalties(self, candidate: Candidate, context: PolicyContext) -> float:
        text = candidate.compact
        score = 0.0

        if candidate.kind == "trainer" and is_miracle_headset(text):
            if context.has_rocket_feather_action or not context.has_non_taunt_attack:
                score += 900.0

        if candidate.kind == "trainer":
            if is_archer(text) and contains_any(text, ("no lost", "cannot", "failed", "不発")):
                score -= 900.0
            if is_factory(text) and context.stadium_right_used:
                score -= 500.0
            if contains_any(text, ("random", "blind")) and context.has_search_action:
                score -= 160.0

        if candidate.kind == "bench" and contains_any(text, ("no bench", "full bench", "ベンチなし", "満員")):
            score -= 1_000.0

        if candidate.kind == "other" and context.has_search_action:
            score -= 120.0

        return score

    def _score_cat_tendency_adjustments(self, candidate: Candidate, context: PolicyContext) -> float:
        text = candidate.compact
        score = 0.0

        # CAT trend: winning samples reached Honchkrow / Rocket Feather earlier.
        if context.turn_number is not None and context.turn_number <= 2:
            if candidate.kind == "evolve" and is_honchkrow(text):
                score += 520.0
            if candidate.kind == "bench" and is_murkrow(text):
                score += 320.0
            if candidate.kind == "trainer" and (is_proton(text) or is_transceiver(text) or is_poke_pad(text)):
                score += 240.0

        # CAT trend: when Feather is available, wandering into draw/search can
        # lose the prize race or deck out. Keep KO pressure ahead of extra value.
        if context.has_rocket_feather_action:
            if candidate.kind == "attack" and is_rocket_feather(text):
                score += 380.0
            if candidate.kind in ("trainer", "ability", "stadium") and not is_miracle_headset(text):
                if context.has_ko_attack:
                    score -= 260.0

        # CAT trend: late deck-out losses were common. With a low deck count,
        # prefer KO/prize actions and avoid non-essential draw.
        if context.self_deck_count is not None and context.self_deck_count <= 6:
            if candidate.kind == "attack" and is_ko_or_prize(text):
                score += 620.0
            if candidate.kind in ("trainer", "ability", "stadium") and contains_any(text, ("draw", "ドロー", "引", "factory", "アテナ")):
                if not context.has_ko_attack:
                    score -= 300.0
                else:
                    score -= 720.0

        # CAT trend: ending with open rights is especially bad for this deck.
        if candidate.kind == "pass":
            if not context.energy_right_used and context.has_energy_action:
                score -= 420.0
            if not context.supporter_right_used and any(is_rocket_supporter(action) for action in context.legal_action_texts):
                score -= 360.0
            if context.has_search_action:
                score -= 260.0

        return score


class MegaExcadrillPolicy:
    """Mega Excadrill / Metal Maker priorities kept separate from global CPU."""

    name = "mega_excadrill"

    deck_terms = (
        "mega excadrill",
        "excadrill",
        "drilbur",
        "beldum",
        "metang",
        "metagross",
        "genesect",
        "metal maker",
        "metal signal",
        "heatran",
        "メガドリュウズ",
        "ドリュウズ",
        "モグリュー",
        "ダンバル",
        "メタング",
        "メタグロス",
        "ゲノセクト",
        "メタルメーカー",
        "メタルシグナル",
        "ヒードラン",
    )

    def matches(self, context: PolicyContext) -> bool:
        if context.profile_name == self.name:
            return True
        return contains_any(context.observation_text, self.deck_terms)

    def score(self, candidate: Candidate, context: PolicyContext) -> float:
        text = candidate.compact
        score = 0.0

        if candidate.kind == "bench":
            if contains_any(text, ("beldum", "ダンバル")):
                score += 620.0
            if contains_any(text, ("drilbur", "モグリュー")):
                score += 520.0
            if contains_any(text, ("genesect", "ゲノセクト", "heatran", "ヒードラン")):
                score += 430.0

        if candidate.kind == "evolve":
            if contains_any(text, ("metang", "メタング")):
                score += 760.0
            if contains_any(text, ("metagross", "メタグロス")):
                score += 560.0
            if contains_any(text, ("mega excadrill", "メガドリュウズ")):
                score += 700.0

        if candidate.kind == "ability":
            if contains_any(text, ("metal signal", "メタルシグナル")):
                score += 840.0
            if contains_any(text, ("metal maker", "メタルメーカー")):
                score += 900.0

        if candidate.kind == "attach_energy":
            if contains_any(text, ("mega excadrill", "excadrill", "drilbur", "メガドリュウズ", "ドリュウズ", "モグリュー")):
                score += 520.0
            if contains_any(text, ("heatran", "ヒードラン")):
                score += 380.0
            if contains_any(text, ("metang", "メタング", "beldum", "ダンバル")):
                score -= 120.0

        if candidate.kind == "switch":
            if contains_any(text, ("metang", "メタング")) and not contains_any(text, ("attack", "ワザ", "metal maker", "メタルメーカー")):
                score -= 520.0
            if contains_any(text, ("mega excadrill", "メガドリュウズ", "heatran", "ヒードラン")):
                score += 300.0

        if candidate.kind == "attack":
            if contains_any(text, ("maximum drill", "マキシマムドリル")):
                score += 540.0
            if contains_any(text, ("dig", "ほりくずす")) and contains_any(text, ("ko", "prize", "きぜつ", "サイド")):
                score += 420.0

        return score


class BossPolicy:
    def __init__(self) -> None:
        self.global_policy = GlobalPokemonPolicy()
        self.deck_policies: tuple[DeckPolicy, ...] = (
            DonkrowPolicy(),
            MegaExcadrillPolicy(),
        )

    def choose(self, observation: Any, information_mode: str = "public", deck_profile: str | None = None) -> int:
        visible = normalize_public_observation(observation) if information_mode != "perfect" else observation
        legal_actions = extract_legal_actions(visible)
        if not legal_actions:
            return 0

        candidates = [
            Candidate(
                index=index,
                raw=action,
                text=(action_text := stringify_candidate_action(action, visible)),
                compact=action_text.lower(),
                kind=classify_action(action_text),
            )
            for index, action in enumerate(legal_actions)
        ]
        profile_name = normalize_profile_name(deck_profile) or detect_deck_profile(visible)
        context = build_policy_context(visible, profile_name, candidates)

        ranked = sorted(
            candidates,
            key=lambda candidate: self.score(candidate, context),
            reverse=True,
        )
        return ranked[0].index

    def score(self, candidate: Candidate, context: PolicyContext) -> float:
        score = self.global_policy.score(candidate, context)
        for deck_policy in self.deck_policies:
            if deck_policy.matches(context):
                score += deck_policy.score(candidate, context)
        return score


DEFAULT_POLICY = BossPolicy()


def choose_action(observation: Any, information_mode: str = "public", deck_profile: str | None = None) -> int:
    return DEFAULT_POLICY.choose(observation, information_mode=information_mode, deck_profile=deck_profile)


def build_policy_context(observation: Any, profile_name: str, candidates: Sequence[Candidate]) -> PolicyContext:
    observation_text = stringify_action(observation).lower()
    legal_action_texts = tuple(candidate.compact for candidate in candidates)
    attack_texts = tuple(candidate.compact for candidate in candidates if candidate.kind == "attack")

    return PolicyContext(
        observation=observation,
        profile_name=profile_name,
        observation_text=observation_text,
        legal_action_texts=legal_action_texts,
        has_attack_action=bool(attack_texts),
        has_non_taunt_attack=any(not is_taunt(text) for text in attack_texts),
        has_ko_attack=any(is_ko_or_prize(text) for text in attack_texts),
        has_rocket_feather_action=any(is_rocket_feather(text) for text in attack_texts),
        has_energy_action=any(candidate.kind == "attach_energy" for candidate in candidates),
        has_bench_action=any(candidate.kind == "bench" for candidate in candidates),
        has_search_action=any(is_search_or_draw_action(candidate) for candidate in candidates),
        energy_right_used=extract_bool_signal(observation, ("energyAttached", "energyAttachedThisTurn", "energy_attached")),
        supporter_right_used=extract_bool_signal(observation, ("supporterUsed", "supporterUsedThisTurn", "supporter_used")),
        stadium_right_used=extract_bool_signal(observation, ("stadiumUsed", "stadiumUsedThisTurn", "stadium_used")),
        self_hand_count=extract_current_player_numeric_signal(observation, ("handCount", "hand_count", "selfHandCount")),
        self_deck_count=extract_current_player_numeric_signal(observation, ("deckCount", "deck_count", "selfDeckCount")),
        self_prize_count=extract_current_player_numeric_signal(observation, ("prizeCount", "prize_count", "selfPrizeCount")),
        opponent_prize_count=extract_numeric_signal_for_role(
            observation,
            ("prizeCount", "prize_count", "opponentPrizeCount", "oppPrizeCount", "enemyPrizeCount"),
            "opponent",
        ),
        turn_number=extract_numeric_signal(observation, ("turn", "turnNumber", "turn_number")),
    )


def normalize_public_observation(observation: Any) -> Any:
    """Return a copy of observation with hidden card identities masked."""

    return sanitize_hidden_info(observation, role="root")


def sanitize_hidden_info(value: Any, role: str) -> Any:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, child in value.items():
            normalized_key = str(key)
            lower_key = normalized_key.lower()
            next_role = role
            if lower_key == "select":
                next_role = "select"
            elif lower_key in OPPONENT_KEYS:
                next_role = "opponent"
            elif lower_key in SELF_KEYS:
                next_role = "self"

            should_mask = (
                role != "select"
                and (
                    (role == "opponent" and lower_key in OPPONENT_HIDDEN_ZONE_KEYS)
                    or (role in ("root", "self") and lower_key in SELF_HIDDEN_ZONE_KEYS)
                )
            )
            if should_mask:
                sanitized[normalized_key] = mask_hidden_zone(child)
            else:
                sanitized[normalized_key] = sanitize_hidden_info(child, next_role)
        return sanitized

    if isinstance(value, list):
        return [sanitize_hidden_info(item, role) for item in value]

    return value


def mask_hidden_zone(value: Any) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [{"hidden": True} for _ in value]
    if isinstance(value, int):
        return value
    return None


def extract_legal_actions(observation: Any) -> Sequence[Any]:
    """Find legal actions from common observation shapes."""

    if isinstance(observation, Mapping):
        select = observation.get("select")
        if isinstance(select, Mapping):
            for key in ("option", "options"):
                value = select.get(key)
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                    return value

        for key in LEGAL_ACTION_KEYS:
            value = observation.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return value

        nested = observation.get("observation")
        if nested is not observation:
            nested_actions = extract_legal_actions(nested)
            if nested_actions:
                return nested_actions

    if isinstance(observation, Sequence) and not isinstance(observation, (str, bytes, bytearray)):
        return observation

    return []


def classify_action(action: Any) -> str:
    text = stringify_action(action).lower()

    if contains_any(text, ("pass", "end", "done", "no_op", "noop", "ターン終了", "番を終")):
        return "pass"
    if contains_any(text, ("ability", "特性", "metal maker", "metal signal", "メタルメーカー", "メタルシグナル")):
        return "ability"
    if contains_any(text, ("attack", "move", "damage", "ワザ", "攻撃", "ダメージ")):
        return "attack"
    if contains_any(text, ("evolve", "evolution", "進化")):
        return "evolve"
    if contains_any(text, ("attach", "energy", "エネルギー", "手張り", "貼")):
        return "attach_energy"
    if contains_any(text, ("stadium", "スタジアム")):
        return "stadium"
    if contains_any(text, ("supporter", "trainer", "item", "search", "draw", "ball", "サポート", "グッズ", "トレーナーズ", "山札", "ドロー", "手札に加")):
        return "trainer"
    if contains_any(text, ("bench", "basic", "play_pokemon", "put_pokemon", "ベンチ", "たね", "場に出")):
        return "bench"
    if contains_any(text, ("retreat", "switch", "promote", "逃げ", "いれかえ", "入れ替え", "前に出")):
        return "switch"
    return "other"


def detect_deck_profile(observation: Any) -> str:
    text = stringify_action(observation).lower()
    if contains_any(text, MegaExcadrillPolicy.deck_terms):
        return "mega_excadrill"
    if contains_any(text, DonkrowPolicy.deck_terms):
        return "donkrow"
    return SUBMISSION_DEFAULT_PROFILE


def normalize_profile_name(value: str | None) -> str | None:
    if not value:
        return None
    compact = value.strip().lower().replace("-", "_").replace(" ", "_")
    if compact in ("rocket_donkrow", "honchkrow", "team_rocket_honchkrow"):
        return "donkrow"
    if compact in ("drill", "excadrill", "mega_drill", "mega_excadrill"):
        return "mega_excadrill"
    if compact in ("default", "global"):
        return "default"
    return compact


def read_field(value: Any, key: str, fallback: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, fallback)
    return getattr(value, key, fallback)


def normalize_option_type(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        try:
            return OPTION_TYPE_NAMES.get(int(normalized), normalized)
        except ValueError:
            return normalized
    if isinstance(value, int):
        return OPTION_TYPE_NAMES.get(value, str(value))
    return str(value)


def normalize_area_name(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("trash", "discard_pile", "discard pile"):
            return "discard"
        try:
            return AREA_TYPE_NAMES.get(int(normalized), normalized)
        except ValueError:
            return normalized
    if isinstance(value, int):
        return AREA_TYPE_NAMES.get(value, f"area{value}")
    return str(value).lower()


def normalize_context_name(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        try:
            return SELECT_CONTEXT_NAMES.get(int(normalized), normalized)
        except ValueError:
            return normalized
    if isinstance(value, int):
        return SELECT_CONTEXT_NAMES.get(value, f"context{value}")
    return str(value).lower()


def card_id(card: Any) -> int | None:
    if isinstance(card, int):
        return card
    value = read_field(card, "id", None)
    if value is None:
        value = read_field(card, "cardId", None)
    if value is None:
        value = read_field(card, "card_id", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def card_name(card: Any) -> str:
    name = read_field(card, "name", None)
    if name:
        return str(name)
    identifier = card_id(card)
    if identifier is not None:
        return CARD_ID_NAMES.get(identifier, f"card-{identifier}")
    return "unknown card"


def card_label(card: Any) -> str:
    identifier = card_id(card)
    if identifier is None:
        return card_name(card)
    return f"{card_name(card)} id:{identifier}"


def card_role(card: Any) -> str:
    identifier = card_id(card)
    if identifier in POKEMON_CARD_IDS:
        if identifier in (463, 473):
            return "basic pokemon bench"
        return "pokemon evolution"
    if identifier in ENERGY_CARD_IDS:
        return "energy"
    if identifier in STADIUM_CARD_IDS:
        return "stadium"
    if identifier in SUPPORTER_CARD_IDS:
        return "supporter trainer"
    if identifier in ITEM_CARD_IDS:
        return "item trainer search"
    return "card"


def observation_current(observation: Any) -> Any:
    return read_field(observation, "current", None)


def observation_select(observation: Any) -> Any:
    return read_field(observation, "select", None)


def current_player_index(current: Any) -> int:
    value = read_field(current, "yourIndex", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def player_state(current: Any, player_index: int | None) -> Any:
    players = read_field(current, "players", None)
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes, bytearray)):
        return None
    if player_index is None:
        player_index = current_player_index(current)
    if 0 <= player_index < len(players):
        return players[player_index]
    return None


def zone_cards(observation: Any, area: Any, player_index: int | None = None) -> Sequence[Any]:
    current = observation_current(observation)
    select = observation_select(observation)
    area_name = normalize_area_name(area)
    player = player_state(current, player_index)

    if area_name == "deck":
        deck = read_field(select, "deck", None)
        if isinstance(deck, Sequence) and not isinstance(deck, (str, bytes, bytearray)):
            return deck
        return []
    if area_name == "looking":
        looking = read_field(current, "looking", None)
        if isinstance(looking, Sequence) and not isinstance(looking, (str, bytes, bytearray)):
            return looking
        return []
    if area_name == "stadium":
        stadium = read_field(current, "stadium", None)
        if isinstance(stadium, Sequence) and not isinstance(stadium, (str, bytes, bytearray)):
            return stadium
        return []

    key = {
        "hand": "hand",
        "discard": "discard",
        "active": "active",
        "bench": "bench",
        "prize": "prize",
    }.get(area_name)
    if key is None:
        return []
    if key == "discard":
        cards: list[Any] = []
        discard = read_field(player, "discard", None)
        trash = read_field(player, "trash", None)
        if isinstance(discard, Sequence) and not isinstance(discard, (str, bytes, bytearray)):
            cards.extend(discard)
        if isinstance(trash, Sequence) and not isinstance(trash, (str, bytes, bytearray)):
            cards.extend(trash)
        return cards
    cards = read_field(player, key, None)
    if isinstance(cards, Sequence) and not isinstance(cards, (str, bytes, bytearray)):
        return cards
    return []


def card_from_area(observation: Any, area: Any, index: Any, player_index: int | None = None) -> Any:
    try:
        normalized_index = int(index)
    except (TypeError, ValueError):
        return None
    cards = zone_cards(observation, area, player_index)
    if 0 <= normalized_index < len(cards):
        return cards[normalized_index]
    return None


def option_card(observation: Any, option: Mapping[str, Any]) -> Any:
    direct_card = read_field(option, "card", None)
    if card_id(direct_card) is not None:
        return direct_card
    direct_id = read_field(option, "cardId", None)
    if direct_id is None:
        direct_id = read_field(option, "card_id", None)
    if direct_id is not None:
        return {"id": direct_id}
    current = observation_current(observation)
    player_index = option.get("playerIndex", current_player_index(current))
    area = option.get("area", 2)
    index = option.get("index")
    return card_from_area(observation, area, index, player_index)


def option_hand_card(observation: Any, option: Mapping[str, Any]) -> Any:
    current = observation_current(observation)
    return card_from_area(observation, 2, option.get("index"), current_player_index(current))


def option_target_text(observation: Any, option: Mapping[str, Any]) -> str:
    if "inPlayArea" not in option:
        return ""
    current = observation_current(observation)
    player_index = option.get("playerIndex", current_player_index(current))
    target = card_from_area(observation, option.get("inPlayArea"), option.get("inPlayIndex"), player_index)
    area_name = normalize_area_name(option.get("inPlayArea"))
    if target is None:
        return f" to {area_name}"
    return f" to {area_name} {card_label(target)}"


def stringify_candidate_action(action: Any, observation: Any) -> str:
    if not isinstance(action, Mapping):
        return stringify_action(action)

    option_type = normalize_option_type(action.get("type", ""))
    option_type_compact = option_type.lower()
    current = observation_current(observation)
    select = observation_select(observation)
    context_text = normalize_context_name(read_field(select, "context", ""))
    effect = read_field(select, "effect", None)
    effect_text = card_label(effect) if effect is not None else ""

    if option_type_compact == "play":
        card = option_hand_card(observation, action)
        role = card_role(card)
        prefix = "bench play pokemon" if "pokemon" in role else f"play {role}"
        return f"{prefix} {card_label(card)} from hand {context_text}"

    if option_type_compact == "attach":
        card = option_hand_card(observation, action)
        role = card_role(card)
        return f"attach {role} {card_label(card)}{option_target_text(observation, action)} {context_text}"

    if option_type_compact == "evolve":
        card = option_hand_card(observation, action)
        return f"evolve {option_target_text(observation, action)} into {card_label(card)} {context_text}"

    if option_type_compact == "ability":
        card = option_card(observation, action)
        return f"ability {card_label(card)} {context_text}"

    if option_type_compact == "attack":
        attack_id = action.get("attackId")
        if attack_id is None:
            attack_id = action.get("attack")
        if attack_id is None:
            normalized_attack_id = -1
        else:
            try:
                normalized_attack_id = int(attack_id)
            except (TypeError, ValueError):
                normalized_attack_id = -1
        attack_name = ATTACK_ID_NAMES.get(normalized_attack_id, f"attack {attack_id}")
        return f"attack {attack_name} {effect_text} {context_text}"

    if option_type_compact == "retreat":
        return f"retreat switch {context_text}"

    if option_type_compact == "end":
        return f"end pass {context_text}"

    if option_type_compact == "card":
        card = option_card(observation, action)
        area_text = normalize_area_name(action.get("area", ""))
        prefix = "choose card"
        if "to hand" in context_text:
            prefix = "search choose to hand"
        elif "setup active" in context_text:
            prefix = "setup active pokemon"
        elif "discard cost" in context_text or card_id(effect) == 891:
            prefix = "rocket feather discard supporter cost choose"
        return f"{prefix} {card_role(card)} {card_label(card)} from {area_text} {effect_text} {context_text}"

    if option_type_compact in ("yes", "no"):
        return option_type

    number = action.get("number")
    if number is not None:
        return f"number {number} {context_text}"

    return stringify_action(action)


def stringify_action(action: Any) -> str:
    if isinstance(action, str):
        return action
    if isinstance(action, Mapping):
        priority_keys = (
            "type",
            "action",
            "name",
            "label",
            "card",
            "card_name",
            "cardName",
            "pokemon",
            "pokemon_name",
            "pokemonName",
            "move",
            "move_name",
            "moveName",
            "target",
            "target_name",
            "targetName",
            "source",
            "from",
            "to",
            "zone",
            "target_zone",
            "targetZone",
            "energy",
            "energy_card",
            "energyCard",
            "cost",
            "effect",
            "reason",
            "description",
        )
        parts: list[str] = []
        for key in priority_keys:
            if key in action:
                parts.append(stringify_action(action[key]))
        if parts:
            return " ".join(part for part in parts if part)
        return json.dumps(action, ensure_ascii=False, sort_keys=True)
    if isinstance(action, Iterable) and not isinstance(action, (str, bytes, bytearray)):
        return " ".join(stringify_action(value) for value in action)
    return str(action)


def contains_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle.lower() in text for needle in needles)


def is_transceiver(text: str) -> bool:
    return contains_any(text, ("team rocket's transceiver", "rocket transceiver", "ロケット団のレシーバー"))


def is_proton(text: str) -> bool:
    return contains_any(text, ("team rocket's proton", "proton", "ロケット団のランス"))


def is_petrel(text: str) -> bool:
    return contains_any(text, ("team rocket's petrel", "petrel", "ロケット団のラムダ"))


def is_ariana(text: str) -> bool:
    return contains_any(text, ("team rocket's ariana", "ariana", "ロケット団のアテナ"))


def is_archer(text: str) -> bool:
    return contains_any(text, ("team rocket's archer", "archer", "ロケット団のアポロ"))


def is_giovanni(text: str) -> bool:
    return contains_any(text, ("team rocket's giovanni", "giovanni", "ロケット団のサカキ"))


def is_rocket_supporter(text: str) -> bool:
    return is_proton(text) or is_petrel(text) or is_ariana(text) or is_archer(text) or is_giovanni(text)


def is_factory(text: str) -> bool:
    return contains_any(text, ("team rocket's factory", "rocket factory", "ロケット団のファクトリー"))


def is_roto_stick(text: str) -> bool:
    return contains_any(text, ("roto-stick", "roto stick", "roto-stick", "ロトりぼう"))


def is_pokegear(text: str) -> bool:
    return contains_any(text, ("pokégear", "pokegear", "pokegear 3.0", "ポケギア"))


def is_poke_pad(text: str) -> bool:
    return contains_any(text, ("poké pad", "poke pad", "ポケパッド"))


def is_night_stretcher(text: str) -> bool:
    return contains_any(text, ("night stretcher", "夜のタンカ"))


def is_miracle_headset(text: str) -> bool:
    return contains_any(text, ("miracle headset", "ミラクルインカム"))


def is_murkrow(text: str) -> bool:
    return contains_any(text, ("team rocket's murkrow", "murkrow", "ロケット団のヤミカラス", "ヤミカラス"))


def is_honchkrow(text: str) -> bool:
    return contains_any(text, ("team rocket's honchkrow", "honchkrow", "donkrow", "ロケット団のドンカラス", "ドンカラス"))


def is_porygon_basic(text: str) -> bool:
    if contains_any(text, ("porygon2", "porygon-z", "porygon z", "ポリゴン2", "ポリゴンz")):
        return False
    return contains_any(text, ("team rocket's porygon", "porygon", "ロケット団のポリゴン", "ポリゴン"))


def is_porygon_attacker(text: str) -> bool:
    return contains_any(text, ("team rocket's porygon2", "team rocket's porygon-z", "porygon2", "porygon-z", "porygon z", "ロケット団のポリゴン2", "ロケット団のポリゴンz", "ポリゴン2", "ポリゴンz"))


def is_porygon_z(text: str) -> bool:
    return contains_any(text, ("porygon-z", "porygon z", "ロケット団のポリゴンz", "ポリゴンz"))


def is_rocket_pokemon(text: str) -> bool:
    return contains_any(text, ("team rocket's", "ロケット団")) and (
        is_murkrow(text)
        or is_honchkrow(text)
        or is_porygon_basic(text)
        or is_porygon_attacker(text)
        or contains_any(text, ("meowth", "persian", "mimikyu", "articuno", "ニャース", "ペルシアン", "ミミッキュ", "フリーザー"))
    )


def is_team_rocket_energy(text: str) -> bool:
    return contains_any(text, ("team rocket's energy", "rocket energy", "ロケット団エネルギー", "ロケット団のエネルギー"))


def is_ignition_energy(text: str) -> bool:
    return contains_any(text, ("ignition energy", "イグニッションエネルギー"))


def is_rocket_feather(text: str) -> bool:
    return contains_any(text, ("rocket feather", "ロケットフェザー"))


def score_rocket_feather_discard_cost(text: str) -> float:
    if not is_rocket_feather(text):
        return 0.0
    if not contains_any(text, ("discard", "trash", "cost", "supporter cost", "trashing", "トラッシュ", "コスト")):
        return 0.0

    score = 0.0
    if is_proton(text):
        score += policy_weight("weights.attack.rocketFeatherCostLance")
    if is_giovanni(text):
        score += policy_weight("weights.attack.rocketFeatherCostGiovanni")
    if is_archer(text):
        score += policy_weight("weights.attack.rocketFeatherCostArcher")
    if is_petrel(text):
        score += policy_weight("weights.attack.rocketFeatherCostPetrel")
    if is_ariana(text):
        score += policy_weight("weights.attack.rocketFeatherCostAthena")
        if contains_any(text, ("no next supporter", "no supporter next turn", "without next supporter", "last supporter", "次ターンサポートなし", "サポートなし", "最後のサポート")):
            score += policy_weight("weights.attack.rocketFeatherCostAthenaNoNextSupporterPenalty")
    return score


def is_taunt(text: str) -> bool:
    return contains_any(text, ("taunt", "いちゃもん"))


def is_ko_or_prize(text: str) -> bool:
    return contains_any(text, ("ko", "knock", "knock out", "prize", "take prize", "サイド", "きぜつ", "気絶", "倒"))


def mentions_active(text: str) -> bool:
    return contains_any(text, ("active", "battle active", "バトル場", "前"))


def mentions_bench(text: str) -> bool:
    return contains_any(text, ("bench", "bench-", "ベンチ", "後ろ"))


def mentions_immediate_attack(text: str) -> bool:
    return contains_any(text, (
        "immediate",
        "this turn",
        "attack now",
        "ready attack",
        "ko now",
        "このターン",
        "即攻撃",
        "今攻撃",
        "ワザ準備",
        "サイドを取る",
    ))


def mentions_future_plan(text: str) -> bool:
    return contains_any(text, (
        "future",
        "next turn",
        "prepare",
        "backup",
        "育成",
        "次の攻撃",
        "次ターン",
        "準備",
        "控え",
    ))


def mentions_proactive_plan(text: str) -> bool:
    return contains_any(text, (
        "proactive",
        "develop",
        "setup",
        "board",
        "展開",
        "盤面",
        "先貼り",
    ))


def is_search_or_draw_action(candidate: Candidate) -> bool:
    text = candidate.compact
    if candidate.kind not in ("trainer", "ability", "stadium"):
        return False
    return contains_any(text, (
        "search",
        "draw",
        "choose",
        "select",
        "deck",
        "hand",
        "ball",
        "transceiver",
        "proton",
        "petrel",
        "gear",
        "pad",
        "roto",
        "factory",
        "山札",
        "ドロー",
        "引",
        "手札に加",
        "レシーバー",
        "ランス",
        "ラムダ",
        "ポケギア",
        "ポケパッド",
        "ロト",
        "ファクトリー",
    ))


def extract_bool_signal(value: Any, keys: Iterable[str]) -> bool:
    wanted = {key.lower() for key in keys}

    def visit(node: Any) -> bool:
        if isinstance(node, Mapping):
            for key, child in node.items():
                if str(key).lower() in wanted and isinstance(child, bool):
                    return child
                if visit(child):
                    return True
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            return any(visit(item) for item in node)
        return False

    return visit(value)


def extract_numeric_signal(value: Any, keys: Iterable[str]) -> int | None:
    wanted = {key.lower() for key in keys}

    def visit(node: Any, role: str = "root") -> int | None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                lower_key = str(key).lower()
                child_role = role
                if lower_key in SELF_KEYS:
                    child_role = "self"
                elif lower_key in OPPONENT_KEYS:
                    child_role = "opponent"
                if lower_key in wanted and isinstance(child, (int, float)) and not isinstance(child, bool):
                    # Prefer explicit self/root counts over opponent counts.
                    if role != "opponent":
                        return int(child)
                nested = visit(child, child_role)
                if nested is not None:
                    return nested
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for item in node:
                nested = visit(item, role)
                if nested is not None:
                    return nested
        return None

    return visit(value)


def extract_current_player_numeric_signal(observation: Any, keys: Iterable[str]) -> int | None:
    current = observation_current(observation)
    player = player_state(current, current_player_index(current))
    value = extract_numeric_signal(player, keys)
    if value is not None:
        return value
    return extract_numeric_signal(observation, keys)


def extract_numeric_signal_for_role(value: Any, keys: Iterable[str], desired_role: str) -> int | None:
    wanted = {key.lower() for key in keys}
    explicit_prefixes = ("opponent", "opp", "enemy") if desired_role == "opponent" else ("self", "me", "player")

    def visit(node: Any, role: str = "root") -> int | None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                lower_key = str(key).lower()
                child_role = role
                if lower_key in SELF_KEYS:
                    child_role = "self"
                elif lower_key in OPPONENT_KEYS:
                    child_role = "opponent"
                if lower_key in wanted and isinstance(child, (int, float)) and not isinstance(child, bool):
                    if role == desired_role or any(lower_key.startswith(prefix) for prefix in explicit_prefixes):
                        return int(child)
                nested = visit(child, child_role)
                if nested is not None:
                    return nested
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for item in node:
                nested = visit(item, role)
                if nested is not None:
                    return nested
        return None

    return visit(value)


def damage_bonus(text: str) -> float:
    numbers = [int(value) for value in re.findall(r"\b\d{1,3}\b", text)]
    if not numbers:
        return 0.0
    return min(max(numbers), 330) / 3.0


def has_non_pass_actions(observation: Any) -> bool:
    return any(classify_action(action) != "pass" for action in extract_legal_actions(observation))


def _has_policy_alias(text: str, alias_key: str) -> bool:
    return contains_any(text, policy_terms(alias_key, ()))


def _install_policy_aliases() -> None:
    global is_transceiver
    global is_proton
    global is_petrel
    global is_ariana
    global is_archer
    global is_giovanni
    global is_rocket_supporter
    global is_factory
    global is_roto_stick
    global is_pokegear
    global is_poke_pad
    global is_night_stretcher
    global is_miracle_headset
    global is_murkrow
    global is_honchkrow
    global is_porygon_basic
    global is_porygon_attacker
    global is_porygon_z
    global is_rocket_pokemon
    global is_team_rocket_energy
    global is_ignition_energy
    global is_rocket_feather
    global is_taunt
    global is_ko_or_prize
    global mentions_active
    global mentions_bench
    global mentions_immediate_attack
    global mentions_future_plan
    global mentions_proactive_plan

    base_is_transceiver = is_transceiver
    base_is_proton = is_proton
    base_is_petrel = is_petrel
    base_is_ariana = is_ariana
    base_is_archer = is_archer
    base_is_giovanni = is_giovanni
    base_is_factory = is_factory
    base_is_roto_stick = is_roto_stick
    base_is_pokegear = is_pokegear
    base_is_poke_pad = is_poke_pad
    base_is_night_stretcher = is_night_stretcher
    base_is_miracle_headset = is_miracle_headset
    base_is_murkrow = is_murkrow
    base_is_honchkrow = is_honchkrow
    base_is_porygon_basic = is_porygon_basic
    base_is_porygon_attacker = is_porygon_attacker
    base_is_porygon_z = is_porygon_z
    base_is_rocket_pokemon = is_rocket_pokemon
    base_is_team_rocket_energy = is_team_rocket_energy
    base_is_ignition_energy = is_ignition_energy
    base_is_rocket_feather = is_rocket_feather
    base_is_taunt = is_taunt
    base_is_ko_or_prize = is_ko_or_prize
    base_mentions_active = mentions_active
    base_mentions_bench = mentions_bench
    base_mentions_immediate_attack = mentions_immediate_attack
    base_mentions_future_plan = mentions_future_plan
    base_mentions_proactive_plan = mentions_proactive_plan

    def synced_is_transceiver(text: str) -> bool:
        return base_is_transceiver(text) or _has_policy_alias(text, "transceiver")

    def synced_is_proton(text: str) -> bool:
        return base_is_proton(text) or _has_policy_alias(text, "proton")

    def synced_is_petrel(text: str) -> bool:
        return base_is_petrel(text) or _has_policy_alias(text, "petrel")

    def synced_is_ariana(text: str) -> bool:
        return base_is_ariana(text) or _has_policy_alias(text, "ariana")

    def synced_is_archer(text: str) -> bool:
        return base_is_archer(text) or _has_policy_alias(text, "archer")

    def synced_is_giovanni(text: str) -> bool:
        return base_is_giovanni(text) or _has_policy_alias(text, "giovanni")

    def synced_is_rocket_supporter(text: str) -> bool:
        return (
            synced_is_proton(text)
            or synced_is_petrel(text)
            or synced_is_ariana(text)
            or synced_is_archer(text)
            or synced_is_giovanni(text)
        )

    def synced_is_factory(text: str) -> bool:
        return base_is_factory(text) or _has_policy_alias(text, "factory")

    def synced_is_roto_stick(text: str) -> bool:
        return base_is_roto_stick(text) or _has_policy_alias(text, "roto_stick")

    def synced_is_pokegear(text: str) -> bool:
        return base_is_pokegear(text) or _has_policy_alias(text, "pokegear")

    def synced_is_poke_pad(text: str) -> bool:
        return base_is_poke_pad(text) or _has_policy_alias(text, "poke_pad")

    def synced_is_night_stretcher(text: str) -> bool:
        return base_is_night_stretcher(text) or _has_policy_alias(text, "night_stretcher")

    def synced_is_miracle_headset(text: str) -> bool:
        return base_is_miracle_headset(text) or _has_policy_alias(text, "miracle_headset")

    def synced_is_murkrow(text: str) -> bool:
        return base_is_murkrow(text) or _has_policy_alias(text, "murkrow")

    def synced_is_honchkrow(text: str) -> bool:
        return base_is_honchkrow(text) or _has_policy_alias(text, "honchkrow")

    def synced_is_porygon_basic(text: str) -> bool:
        if base_is_porygon_basic(text):
            return True
        if _has_policy_alias(text, "porygon_attacker") or _has_policy_alias(text, "porygon_z"):
            return False
        return _has_policy_alias(text, "porygon_basic")

    def synced_is_porygon_attacker(text: str) -> bool:
        return base_is_porygon_attacker(text) or _has_policy_alias(text, "porygon_attacker")

    def synced_is_porygon_z(text: str) -> bool:
        return base_is_porygon_z(text) or _has_policy_alias(text, "porygon_z")

    def synced_is_rocket_pokemon(text: str) -> bool:
        return base_is_rocket_pokemon(text) or _has_policy_alias(text, "rocket_pokemon") or (
            _has_policy_alias(text, "team_rocket")
            and (
                synced_is_murkrow(text)
                or synced_is_honchkrow(text)
                or synced_is_porygon_basic(text)
                or synced_is_porygon_attacker(text)
            )
        )

    def synced_is_team_rocket_energy(text: str) -> bool:
        return base_is_team_rocket_energy(text) or _has_policy_alias(text, "team_rocket_energy")

    def synced_is_ignition_energy(text: str) -> bool:
        return base_is_ignition_energy(text) or _has_policy_alias(text, "ignition_energy")

    def synced_is_rocket_feather(text: str) -> bool:
        return base_is_rocket_feather(text) or _has_policy_alias(text, "rocket_feather")

    def synced_is_taunt(text: str) -> bool:
        return base_is_taunt(text) or _has_policy_alias(text, "taunt")

    def synced_is_ko_or_prize(text: str) -> bool:
        return base_is_ko_or_prize(text) or _has_policy_alias(text, "ko_or_prize")

    def synced_mentions_active(text: str) -> bool:
        return base_mentions_active(text) or _has_policy_alias(text, "active")

    def synced_mentions_bench(text: str) -> bool:
        return base_mentions_bench(text) or _has_policy_alias(text, "bench")

    def synced_mentions_immediate_attack(text: str) -> bool:
        return base_mentions_immediate_attack(text) or _has_policy_alias(text, "immediate_attack")

    def synced_mentions_future_plan(text: str) -> bool:
        return base_mentions_future_plan(text) or _has_policy_alias(text, "future_plan")

    def synced_mentions_proactive_plan(text: str) -> bool:
        return base_mentions_proactive_plan(text) or _has_policy_alias(text, "proactive_plan")

    is_transceiver = synced_is_transceiver
    is_proton = synced_is_proton
    is_petrel = synced_is_petrel
    is_ariana = synced_is_ariana
    is_archer = synced_is_archer
    is_giovanni = synced_is_giovanni
    is_rocket_supporter = synced_is_rocket_supporter
    is_factory = synced_is_factory
    is_roto_stick = synced_is_roto_stick
    is_pokegear = synced_is_pokegear
    is_poke_pad = synced_is_poke_pad
    is_night_stretcher = synced_is_night_stretcher
    is_miracle_headset = synced_is_miracle_headset
    is_murkrow = synced_is_murkrow
    is_honchkrow = synced_is_honchkrow
    is_porygon_basic = synced_is_porygon_basic
    is_porygon_attacker = synced_is_porygon_attacker
    is_porygon_z = synced_is_porygon_z
    is_rocket_pokemon = synced_is_rocket_pokemon
    is_team_rocket_energy = synced_is_team_rocket_energy
    is_ignition_energy = synced_is_ignition_energy
    is_rocket_feather = synced_is_rocket_feather
    is_taunt = synced_is_taunt
    is_ko_or_prize = synced_is_ko_or_prize
    mentions_active = synced_mentions_active
    mentions_bench = synced_mentions_bench
    mentions_immediate_attack = synced_mentions_immediate_attack
    mentions_future_plan = synced_mentions_future_plan
    mentions_proactive_plan = synced_mentions_proactive_plan


_install_policy_aliases()


def rank_actions(observation: Any, information_mode: str = "public", deck_profile: str | None = None) -> list[dict[str, Any]]:
    """Debug helper for local tests and future CAT-style score inspection."""

    visible = normalize_public_observation(observation) if information_mode != "perfect" else observation
    candidates = [
        Candidate(
            index=index,
            raw=action,
            text=(action_text := stringify_candidate_action(action, visible)),
            compact=action_text.lower(),
            kind=classify_action(action_text),
        )
        for index, action in enumerate(extract_legal_actions(visible))
    ]
    profile_name = normalize_profile_name(deck_profile) or detect_deck_profile(visible)
    context = build_policy_context(visible, profile_name, candidates)
    return [
        {
            "index": candidate.index,
            "kind": candidate.kind,
            "score": DEFAULT_POLICY.score(candidate, context),
            "text": candidate.text,
        }
        for candidate in sorted(candidates, key=lambda item: DEFAULT_POLICY.score(item, context), reverse=True)
    ]



# Donkrow submission adapter and structured action overrides.

HONCHKROW = 891
MURKROW = 463
PORYGON = 473
PORYGON2 = 474
TEAM_ROCKET_ENERGY = 15
IGNITION_ENERGY = 17
ROCKET_FEATHER_ATTACK = 1285
TAUNT_ATTACK = 1286
PORYGON2_R_COMMAND_ATTACKS = {670}
MURKROW_TEMPT_ATTACK = 652
MURKROW_SECONDARY_ATTACK = 653
MURKROW_ATTACKS = {MURKROW_TEMPT_ATTACK, MURKROW_SECONDARY_ATTACK}
ROCKET_SUPPORTER_COST_PRIORITY = {
    1220: 100,  # Team Rocket's Proton / Lance: lowest future value after setup
    1218: 90,  # Team Rocket's Giovanni / Sakaki
    1217: 80,  # Team Rocket's Archer / Apollo
    1219: 70,  # Team Rocket's Petrel / Lambda
    1216: -250,  # Team Rocket's Ariana / Athena: preserve if possible
}
ROCKET_SUPPORTERS = set(ROCKET_SUPPORTER_COST_PRIORITY)
BASIC_SETUP_POKEMON = {MURKROW, PORYGON}
ROCKET_FIELD_POKEMON = {MURKROW, HONCHKROW, PORYGON, PORYGON2}
POKEGEAR = 1122
TEAM_ROCKET_TRANSCEIVER = 1134
ROTO_STICK = 1077
POKE_PAD = 1152
NIGHT_STRETCHER = 1097
MIRACLE_HEADSET = 1109
PROTON = 1220
PETREL = 1219
GIOVANNI = 1218
ARCHER = 1217
ARIANA = 1216
FACTORY = 1257
ABRA = 741
KADABRA = 742
ALAKAZAM = 743
DARK_WEAK_TO_IDS = {ABRA, KADABRA, ALAKAZAM, 431, 741, 742, 743, 770, 771, 772, 1059}
DARK_WEAK_NAME_HINTS = (
    "abra",
    "kadabra",
    "alakazam",
    "gastly",
    "haunter",
    "gengar",
    "mewtwo",
)
TWO_PRIZE_POKEMON_IDS = {
    24, 29, 30, 37, 40, 44, 46, 52, 63, 75, 79, 80, 83, 84, 96, 99,
    107, 108, 117, 121, 125, 130, 138, 139, 140, 141, 150, 153, 154, 161, 176, 179,
    184, 189, 190, 193, 198, 205, 207, 210, 223, 229, 231, 232, 236, 239, 241, 243,
    244, 246, 248, 249, 259, 269, 272, 283, 293, 299, 302, 306, 313, 316, 320, 326,
    328, 329, 331, 336, 337, 340, 357, 369, 372, 381, 389, 404, 407, 424, 431, 447,
    455, 458, 471, 481, 509, 515, 525, 527, 547, 561, 573, 583, 598, 618, 631, 641,
    648, 652, 662, 678, 687, 695, 723, 737, 747, 754, 756, 766, 772, 781, 790, 795,
    806, 813, 828, 835, 849, 861, 868, 886, 896, 904, 911, 919, 928, 932, 939, 944,
    951, 954, 957, 962, 968, 969, 975, 979, 984, 988, 990, 993, 997, 1002, 1006, 1022,
    1026, 1031, 1040, 1056, 1062, 1064, 1071, 1145,
}


def load_deck():
    candidates = (
        Path("deck.csv"),
        Path("/kaggle_simulations/agent/deck.csv"),
    )
    for deck_path in candidates:
        if deck_path.exists():
            deck = [int(line.strip()) for line in deck_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(deck) != 60:
                raise ValueError(f"deck.csv must contain exactly 60 cards, got {len(deck)}")
            return deck
    raise FileNotFoundError("deck.csv was not found")


MY_DECK = load_deck()


def _policy_rule_enabled(rule_id, fallback=True):
    value = policy_value(f"rules.{rule_id}.enabled", fallback)
    return value if isinstance(value, bool) else fallback


def _policy_rule_number(rule_id, key, fallback):
    value = policy_value(f"rules.{rule_id}.{key}", fallback)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return value


def _policy_rule_bool(rule_id, key, fallback):
    value = policy_value(f"rules.{rule_id}.{key}", fallback)
    return value if isinstance(value, bool) else fallback


def _select_payload(observation):
    if isinstance(observation, dict):
        return observation.get("select")
    return getattr(observation, "select", None)


def _select_options(select):
    if select is None:
        return []
    if isinstance(select, dict):
        value = select.get("option")
        if value is None:
            value = select.get("options")
        return value if isinstance(value, list) else []
    value = getattr(select, "option", None)
    if value is None:
        value = getattr(select, "options", None)
    return value if isinstance(value, list) else []


def _selection_bounds(select):
    if select is None:
        return 0, 0
    if isinstance(select, dict):
        min_count = select.get("minCount", select.get("min_count", 1))
        max_count = select.get("maxCount", select.get("max_count", 1))
    else:
        min_count = getattr(select, "minCount", getattr(select, "min_count", 1))
        max_count = getattr(select, "maxCount", getattr(select, "max_count", 1))
    try:
        min_count = int(min_count)
    except (TypeError, ValueError):
        min_count = 1
    try:
        max_count = int(max_count)
    except (TypeError, ValueError):
        max_count = 1
    return max(0, min_count), max(0, max_count)


def _read(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _card_id(card):
    if isinstance(card, int):
        return card
    if card is None:
        return None
    value = _read(card, "id")
    if value is None:
        value = _read(card, "cardId")
    if value is None:
        value = _read(card, "card_id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _iter_cards(value):
    if isinstance(value, list):
        for item in value:
            yield from _iter_cards(item)
        return
    if _card_id(value) is not None:
        yield value


def _top_card(value):
    if isinstance(value, list):
        for item in reversed(value):
            top = _top_card(item)
            if top is not None:
                return top
        return None
    return value if _card_id(value) is not None else None


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _select_context(select):
    value = _read(select, "context")
    if isinstance(value, str):
        normalized = value.strip().lower()
        try:
            return int(normalized)
        except ValueError:
            return normalized
    return _safe_int(value, -1)


def _option_type(option):
    value = _read(option, "type")
    if isinstance(value, str):
        normalized = value.strip().lower()
        try:
            return int(normalized)
        except ValueError:
            return normalized
    return _safe_int(value, -1)


def _area_code(value, default=2):
    if isinstance(value, str):
        normalized = value.strip().lower()
        area_names = {
            "deck": 1,
            "hand": 2,
            "discard": 3,
            "trash": 3,
            "active": 4,
            "bench": 5,
            "prize": 6,
            "prizes": 6,
            "stadium": 7,
            "looking": 12,
            "look": 12,
        }
        if normalized in area_names:
            return area_names[normalized]
        try:
            return int(normalized)
        except ValueError:
            return default
    return _safe_int(value, default)


def _player_state(current, index):
    players = _read(current, "players", [])
    if isinstance(players, list) and 0 <= index < len(players):
        return players[index]
    return {}


def _zone_cards(current, select, area, player_index):
    area = _area_code(area, -1)
    if area == 1:
        deck = _read(select, "deck", [])
        return deck if isinstance(deck, list) else []
    if area == 12:
        looking = _read(current, "looking", [])
        return looking if isinstance(looking, list) else []
    if area == 7:
        stadium = _read(current, "stadium", [])
        return stadium if isinstance(stadium, list) else []

    player = _player_state(current, player_index)
    key = {
        2: "hand",
        3: "discard",
        4: "active",
        5: "bench",
        6: "prize",
    }.get(area)
    if key == "discard":
        discard = _read(player, "discard", [])
        trash = _read(player, "trash", [])
        cards = []
        if isinstance(discard, list):
            cards.extend(discard)
        if isinstance(trash, list):
            cards.extend(trash)
        return cards
    cards = _read(player, key, []) if key else []
    return cards if isinstance(cards, list) else []


def _card_from_option(observation, option):
    direct_card = _read(option, "card")
    if _card_id(direct_card) is not None:
        return direct_card
    for key in ("cardId", "card_id"):
        direct_id = _card_id({ "id": _read(option, key) })
        if direct_id is not None:
            return {"id": direct_id}

    current = _read(observation, "current", {})
    select = _read(observation, "select", {})
    your_index = _safe_int(_read(current, "yourIndex"), 0)
    player_index = _safe_int(_read(option, "playerIndex"), your_index)
    area = _area_code(_read(option, "area"), 2)
    index = _safe_int(_read(option, "index"), -1)
    cards = _zone_cards(current, select, area, player_index)
    if 0 <= index < len(cards):
        return _top_card(cards[index])
    return None


def _target_card_from_option(observation, option):
    area = _read(option, "inPlayArea")
    index = _read(option, "inPlayIndex")
    if area is None or index is None:
        return None
    current = _read(observation, "current", {})
    select = _read(observation, "select", {})
    your_index = _safe_int(_read(current, "yourIndex"), 0)
    player_index = _safe_int(_read(option, "playerIndex"), your_index)
    cards = _zone_cards(current, select, _area_code(area, -1), player_index)
    normalized_index = _safe_int(index, -1)
    if 0 <= normalized_index < len(cards):
        return _top_card(cards[normalized_index])
    return None


def _promotion_card_from_option(observation, option):
    if _option_type(option) not in (7, "play", 12, "retreat", "switch", "promote"):
        return None
    if _read(option, "area") is not None or _read(option, "inPlayArea") is not None:
        return None
    current = _read(observation, "current", {})
    select = _read(observation, "select", {})
    your_index = _safe_int(_read(current, "yourIndex"), 0)
    player_index = _safe_int(_read(option, "playerIndex"), your_index)
    if player_index != your_index:
        return None
    player = _player_state(current, player_index)
    if _zone_count(player, "active") > 0:
        return None
    index = _safe_int(_read(option, "index"), -1)
    bench = _zone_cards(current, select, 5, player_index)
    if 0 <= index < len(bench):
        return _top_card(bench[index])
    return None


def _current_player(observation):
    current = _read(observation, "current", {})
    your_index = _safe_int(_read(current, "yourIndex"), 0)
    return current, your_index, _player_state(current, your_index)


def _zone_count(player, key):
    if key == "discard":
        cards = []
        discard = _read(player, "discard", [])
        trash = _read(player, "trash", [])
        if isinstance(discard, list):
            cards.extend(discard)
        if isinstance(trash, list):
            cards.extend(trash)
    else:
        cards = _read(player, key, [])
    return sum(1 for _ in _iter_cards(cards))


def _hand_count(player):
    explicit = _read(player, "handCount")
    if explicit is not None:
        return _safe_int(explicit, 0)
    return _zone_count(player, "hand")


def _field_card_ids(player):
    ids = []
    for zone in ("active", "bench"):
        for card in _iter_cards(_read(player, zone, [])):
            identifier = _card_id(card)
            if identifier is not None:
                ids.append(identifier)
    return ids


def _field_cards(player):
    cards = []
    for zone in ("active", "bench"):
        cards.extend(_iter_cards(_read(player, zone, [])))
    return cards


def _field_top_cards(player):
    cards = []
    active = _top_card(_read(player, "active", []))
    if active is not None:
        cards.append(active)
    bench = _read(player, "bench", [])
    if isinstance(bench, list):
        for slot in bench:
            top = _top_card(slot)
            if top is not None:
                cards.append(top)
    return cards


def _bench_top_count(player):
    bench = _read(player, "bench", [])
    if not isinstance(bench, list):
        return 0
    return sum(1 for slot in bench if _top_card(slot) is not None)


def _bench_limit(player):
    return max(0, _safe_int(_read(player, "benchMax"), 5))


def _is_basic_setup_pokemon(identifier):
    return identifier in BASIC_SETUP_POKEMON


def _basic_setup_ids_in_hand(player):
    return [
        _card_id(card)
        for card in _iter_cards(_read(player, "hand", []))
        if _is_basic_setup_pokemon(_card_id(card))
    ]


def _needs_seed_out_bench_guard(player):
    field_count = len(_field_top_cards(player))
    bench_count = _bench_top_count(player)
    return field_count <= 1 and bench_count < _bench_limit(player)


def _needs_thin_board_basic_guard(player):
    field_count = len(_field_top_cards(player))
    bench_count = _bench_top_count(player)
    return field_count <= 2 and bench_count < _bench_limit(player)


def _has_basic_play_option(observation):
    select = _read(observation, "select", {})
    for option in _select_options(select):
        if _option_type(option) in (7, "play"):
            identifier = _card_id(_card_from_option(observation, option))
            if _is_basic_setup_pokemon(identifier):
                return True
    return False


def _has_basic_continuity_option(observation):
    if _has_basic_play_option(observation):
        return True
    select = _read(observation, "select", {})
    for option in _select_options(select):
        if _option_type(option) not in (7, "play"):
            continue
        identifier = _card_id(_card_from_option(observation, option))
        if identifier in (POKE_PAD, NIGHT_STRETCHER, PROTON, TEAM_ROCKET_TRANSCEIVER):
            return True
    return False


def _is_ariana_compression_option(observation, option):
    option_type = _option_type(option)
    identifier = _card_id(_card_from_option(observation, option))
    if option_type in (9, "evolve"):
        _, _, player = _current_player(observation)
        if identifier == HONCHKROW:
            return True
        if identifier == PORYGON2:
            return _porygon_development_allowed(player)
        return False
    if option_type not in (7, "play"):
        return False
    if identifier in (MURKROW, PORYGON, HONCHKROW, PORYGON2):
        return True
    if identifier in (POKE_PAD, POKEGEAR, ROTO_STICK):
        return True
    if identifier == TEAM_ROCKET_TRANSCEIVER:
        return True
    return False


def _seed_out_basic_play_score(observation, option):
    _, _, player = _current_player(observation)
    if not _needs_seed_out_bench_guard(player):
        return None
    if _option_type(option) not in (7, "play"):
        return None
    identifier = _card_id(_card_from_option(observation, option))
    if identifier == MURKROW:
        return 90_000
    if identifier == PORYGON:
        return 72_000
    return None


def _rocket_feather_damage_per_supporter(opponent_active):
    defender_id = _card_id(opponent_active)
    defender_name = str(_read(opponent_active, "name", "")).lower()
    dark_weak_by_name = any(hint in defender_name for hint in DARK_WEAK_NAME_HINTS)
    return 120 if defender_id in DARK_WEAK_TO_IDS or dark_weak_by_name else 60


def _night_stretcher_target_score(observation, card):
    identifier = _card_id(card)
    if identifier is None:
        return -100_000

    current, your_index, player = _current_player(observation)
    field_top_ids = [_card_id(field_card) for field_card in _field_top_cards(player)]
    hand_ids = [_card_id(hand_card) for hand_card in _iter_cards(_read(player, "hand", []))]
    bench_count = _bench_top_count(player)
    field_count = len(field_top_ids)
    free_bench = bench_count < _bench_limit(player)
    seed_out_risk = field_count <= 1 and free_bench
    thin_board_basic_risk = field_count <= 2 and free_bench
    has_murkrow_source = MURKROW in field_top_ids or MURKROW in hand_ids
    has_porygon_source = PORYGON in field_top_ids or PORYGON in hand_ids
    has_honchkrow_ready = HONCHKROW in field_top_ids or HONCHKROW in hand_ids
    has_porygon2_ready = PORYGON2 in field_top_ids or PORYGON2 in hand_ids

    if _is_basic_setup_pokemon(identifier):
        if seed_out_risk:
            return 300_000 + (24_000 if identifier == MURKROW else 12_000)
        if thin_board_basic_risk:
            return 245_000 + (24_000 if identifier == MURKROW else 12_000)
        score = 118_000 if identifier == MURKROW else 92_000
        if free_bench:
            score += 34_000
        if identifier == MURKROW and not has_honchkrow_ready:
            score += 16_000
        if identifier == PORYGON and not has_porygon2_ready:
            score += 10_000
        return score

    if seed_out_risk or thin_board_basic_risk:
        return -85_000
    if identifier == HONCHKROW:
        return 180_000 if has_murkrow_source else 22_000
    if identifier == PORYGON2:
        if not _porygon_development_allowed(player):
            return 12_000
        return 165_000 if has_porygon_source else 18_000
    if identifier in ROCKET_FIELD_POKEMON:
        return 40_000
    if identifier in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY):
        return 16_000
    return -20_000


def _best_night_stretcher_target_score(observation):
    _, _, player = _current_player(observation)
    targets = []
    for zone in ("discard", "trash"):
        for card in _iter_cards(_read(player, zone, [])):
            targets.append(card)
    if not targets:
        return -100_000
    return max(_night_stretcher_target_score(observation, card) for card in targets)


def _poke_pad_target_score(observation, card):
    identifier = _card_id(card)
    if identifier is None:
        return -100_000

    _, _, player = _current_player(observation)
    field_top_ids = [_card_id(field_card) for field_card in _field_top_cards(player)]
    hand_ids = [_card_id(hand_card) for hand_card in _iter_cards(_read(player, "hand", []))]
    bench_count = _bench_top_count(player)
    field_count = len(field_top_ids)
    free_bench = bench_count < _bench_limit(player)
    seed_out_risk = field_count <= 1 and free_bench
    thin_board_basic_risk = field_count <= 2 and free_bench
    has_murkrow_source = MURKROW in field_top_ids or MURKROW in hand_ids
    has_porygon_source = PORYGON in field_top_ids or PORYGON in hand_ids
    has_honchkrow_ready = HONCHKROW in field_top_ids or HONCHKROW in hand_ids
    has_porygon2_ready = PORYGON2 in field_top_ids or PORYGON2 in hand_ids

    if _is_basic_setup_pokemon(identifier):
        if seed_out_risk:
            return 300_000 + (24_000 if identifier == MURKROW else 12_000)
        if thin_board_basic_risk:
            return 245_000 + (24_000 if identifier == MURKROW else 12_000)
        score = 118_000 if identifier == MURKROW else 92_000
        if free_bench:
            score += 34_000
        if identifier == MURKROW and not has_honchkrow_ready:
            score += 16_000
        if identifier == PORYGON and not has_porygon2_ready:
            score += 10_000
        return score

    if seed_out_risk or thin_board_basic_risk:
        return -85_000
    if identifier == HONCHKROW:
        return 180_000 if has_murkrow_source else 22_000
    if identifier == PORYGON2:
        if not _porygon_development_allowed(player):
            return 12_000
        return 165_000 if has_porygon_source else 18_000
    if identifier in ROCKET_FIELD_POKEMON:
        return 40_000
    return -20_000


def _best_poke_pad_target_score(observation):
    _, _, player = _current_player(observation)
    targets = []
    for card in _iter_cards(_read(player, "deck", [])):
        identifier = _card_id(card)
        if identifier in ROCKET_FIELD_POKEMON:
            targets.append(card)
    if not targets:
        return -100_000
    return max(_poke_pad_target_score(observation, card) for card in targets)


def _recovery_tool_use_score(target_score):
    if target_score >= 250_000:
        return 92_000
    if target_score >= 150_000:
        return 72_000
    if target_score >= 75_000:
        return 28_000
    if target_score >= 35_000:
        return 9_500
    return -7_000


def _energy_prepared_murkrow_needs_honchkrow(player, hand_ids, deck_ids):
    if HONCHKROW in hand_ids or HONCHKROW not in deck_ids:
        return False
    for card in _field_top_cards(player):
        if _card_id(card) == MURKROW and _attached_energy_cards(card) > 0:
            return True
    return False


def _count_cards(player, zones, predicate):
    total = 0
    for zone in zones:
        cards = []
        if zone == "discard":
            discard = _read(player, "discard", [])
            trash = _read(player, "trash", [])
            if isinstance(discard, list):
                cards.extend(discard)
            if isinstance(trash, list):
                cards.extend(trash)
        else:
            cards = _read(player, zone, [])
        for card in _iter_cards(cards):
            if predicate(_card_id(card)):
                total += 1
    return total


def _effect_id(observation):
    select = _read(observation, "select", {})
    return _card_id(_read(select, "effect"))


def _attached_energy_units(card):
    energies = _read(card, "energies", [])
    if isinstance(energies, list):
        return len(energies)
    energy_cards = _read(card, "energyCards", [])
    if isinstance(energy_cards, list):
        return len(energy_cards)
    return 0


def _attached_energy_cards(card):
    energy_cards = _read(card, "energyCards", [])
    if isinstance(energy_cards, list):
        return len(energy_cards)
    energies = _read(card, "energies", [])
    if isinstance(energies, list):
        return len(energies)
    return 0


def _attached_energy_card_ids(card):
    ids = []
    energy_cards = _read(card, "energyCards", [])
    if isinstance(energy_cards, list):
        for energy_card in energy_cards:
            identifier = _card_id(energy_card)
            if identifier is not None:
                ids.append(identifier)
    energies = _read(card, "energies", [])
    if isinstance(energies, list):
        for energy in energies:
            identifier = _card_id(energy)
            if identifier is None and isinstance(energy, int):
                identifier = energy
            if identifier is not None:
                ids.append(identifier)
    return ids


def _attack_id(option):
    for key in ("attackId", "attack", "moveId", "move_id"):
        value = _read(option, key)
        if value is not None:
            return _safe_int(value, -1)
    return -1


def _deck_count(player):
    explicit = _read(player, "deckCount")
    if explicit is not None:
        return _safe_int(explicit, 0)
    deck = _read(player, "deck", [])
    return len(deck) if isinstance(deck, list) else 0


def _prize_cards_remaining(player):
    explicit = _read(player, "prizeCount", _read(player, "prize_count"))
    if explicit is not None:
        parsed = _safe_int(explicit, 0)
        if parsed > 0:
            return parsed
    prize = _read(player, "prize", _read(player, "prizes", []))
    if isinstance(prize, list):
        return len(prize)
    return _zone_count(player, "prize")


def _remaining_hp(card):
    if card is None:
        return 0
    for key in ("hp", "remainingHp", "remainingHP", "currentHp", "currentHP"):
        value = _read(card, key)
        if value is not None:
            parsed = _safe_int(value, 0)
            if parsed > 0:
                return parsed
    max_hp = _safe_int(_read(card, "maxHp", _read(card, "maxHP")), 0)
    damage = _safe_int(
        _read(card, "damage", _read(card, "damageCounters", _read(card, "damageCounter"))),
        0,
    )
    if max_hp > 0:
        return max(0, max_hp - damage)
    return 0


def _opponent_active_card(observation):
    current = _read(observation, "current", {})
    your_index = _safe_int(_read(current, "yourIndex"), 0)
    opponent = _player_state(current, 1 - your_index)
    return _top_card(_read(opponent, "active", []))


def _opponent_bench_cards(observation):
    current = _read(observation, "current", {})
    your_index = _safe_int(_read(current, "yourIndex"), 0)
    opponent = _player_state(current, 1 - your_index)
    bench = _read(opponent, "bench", [])
    if not isinstance(bench, list):
        return []
    return [top for slot in bench if (top := _top_card(slot)) is not None]


def _opponent_hand_count(observation):
    current = _read(observation, "current", {})
    your_index = _safe_int(_read(current, "yourIndex"), 0)
    opponent = _player_state(current, 1 - your_index)
    for source in (opponent, current):
        for key in ("handCount", "hand_count", "opponentHandCount", "oppHandCount", "enemyHandCount"):
            value = _read(source, key)
            if value is not None:
                return _safe_int(value, 0)
    hand = _read(opponent, "hand", [])
    return _zone_count(opponent, "hand") if isinstance(hand, list) else 0


def _bool_signal(sources, keys):
    for source in sources:
        for key in keys:
            value = _read(source, key)
            if value is not None:
                if isinstance(value, str):
                    normalized = value.strip().lower()
                    if normalized in ("false", "0", "no", "none", "null"):
                        return False
                    if normalized in ("true", "1", "yes"):
                        return True
                return bool(value)
    return None


def _logs_indicate_lost_rocket_mon_last_turn(observation, your_index):
    logs = _read(observation, "logs", [])
    if not isinstance(logs, list):
        return None
    for entry in logs:
        player_index = _safe_int(_read(entry, "playerIndex"), -1)
        if player_index != your_index:
            continue
        identifier = _card_id({"id": _read(entry, "cardId", _read(entry, "card_id"))})
        if identifier not in ROCKET_FIELD_POKEMON:
            continue
        from_area = _area_code(_read(entry, "fromArea", _read(entry, "from_area")), -1)
        to_area = _area_code(_read(entry, "toArea", _read(entry, "to_area")), -1)
        if from_area in (4, 5) and to_area == 3:
            return True
    return None


def _lost_rocket_mon_last_turn(observation, current, player):
    rocket_signal = _bool_signal(
        (player, current),
        (
            "lostRocketMonLastTurn",
            "lost_rocket_mon_last_turn",
            "lostRocketPokemonLastTurn",
            "rocketPokemonLostLastTurn",
            "rocketPokemonKnockedOutLastTurn",
        ),
    )
    if rocket_signal is not None:
        return rocket_signal
    generic_signal = _bool_signal((player, current), ("lostMonLastTurn", "lost_mon_last_turn", "pokemonLostLastTurn"))
    if generic_signal is not None:
        return generic_signal
    your_index = _safe_int(_read(current, "yourIndex"), 0)
    return _logs_indicate_lost_rocket_mon_last_turn(observation, your_index)


def _card_name_text(card):
    chunks = []
    for key in ("name", "cardName", "card_name", "englishName", "english_name", "pokemonName", "pokemon_name"):
        value = _read(card, key)
        if value:
            chunks.append(str(value))
    identifier = _card_id(card)
    if identifier is not None and identifier in CARD_ID_NAMES:
        chunks.append(CARD_ID_NAMES[identifier])
    return " ".join(chunks).lower()


def _text_from_card_field(card, key):
    value = _read(card, key)
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    if isinstance(value, dict):
        return " ".join(str(item) for item in value.values())
    return "" if value is None else str(value)


def _prize_count_for_knockout(card):
    for key in (
        "prizeCount",
        "prize_count",
        "knockoutPrizeCount",
        "knockout_prize_count",
        "prizesTaken",
        "prizeValue",
    ):
        value = _read(card, key)
        if value is not None:
            parsed = _safe_int(value, 0)
            if parsed > 0:
                return parsed

    identifier = _card_id(card)
    if identifier in TWO_PRIZE_POKEMON_IDS:
        return 2

    text = " ".join(
        [
            _card_name_text(card),
            _text_from_card_field(card, "rule"),
            _text_from_card_field(card, "rules"),
            _text_from_card_field(card, "ruleBox"),
            _text_from_card_field(card, "rule_box"),
            _text_from_card_field(card, "subtypes"),
            _text_from_card_field(card, "tags"),
        ]
    ).lower()
    if re.search(r"\bmega\b|\bex\b|ex$", text):
        return 2
    return 1


@dataclass(frozen=True)
class DonkrowTurnPlan:
    attacker_card: Any
    attacker_id: int
    attacker_index: int
    target_card: Any
    target_index: int
    attack_id: int
    damage: int
    remaining_hp: int
    prize_count: int
    score: int
    can_ko: bool
    needs_energy: bool
    needs_switch: bool
    game_end: bool
    seed_guard_blocked: bool


def _same_card_instance(left, right):
    if left is None or right is None:
        return False
    left_serial = _read(left, "serial")
    right_serial = _read(right, "serial")
    if left_serial is not None and right_serial is not None:
        return left_serial == right_serial and _card_id(left) == _card_id(right)
    if isinstance(left, dict) and isinstance(right, dict) and left == right:
        return True
    return False


def _option_targets_plan_attacker(target, plan):
    return plan is not None and _same_card_instance(target, plan.attacker_card)


def _switch_option_targets_plan_attacker(option, target, promotion_card, plan):
    if plan is None:
        return False
    if _option_targets_plan_attacker(target, plan):
        return True
    if _option_targets_plan_attacker(promotion_card, plan):
        return True
    option_index = _safe_int(_read(option, "index"), -1)
    in_play_index = _safe_int(_read(option, "inPlayIndex"), -1)
    return option_index == plan.attacker_index - 1 or in_play_index == plan.attacker_index - 1


def _planned_attack_damage(player, attacker, target, attack_id):
    attacker_id = _card_id(attacker)
    if attacker_id == HONCHKROW and attack_id == ROCKET_FEATHER_ATTACK:
        hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)
        if hand_supporters <= 0:
            return 0
        return _rocket_feather_damage_per_supporter(target) * min(4, hand_supporters)
    if attacker_id == PORYGON2 and attack_id in PORYGON2_R_COMMAND_ATTACKS:
        discard_supporters = _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS)
        return discard_supporters * _policy_rule_number("porygon2RCommandFallback", "damagePerTrashRocketSupporter", 20)
    if attacker_id == MURKROW and attack_id in MURKROW_ATTACKS:
        return 0
    return 0


def _planned_attack_ids(attacker):
    attacker_id = _card_id(attacker)
    if attacker_id == HONCHKROW:
        return (ROCKET_FEATHER_ATTACK,)
    if attacker_id == PORYGON2:
        return tuple(PORYGON2_R_COMMAND_ATTACKS)
    if attacker_id == MURKROW:
        return ()
    return ()


def _planned_attacker_needs_energy(attacker, attack_id):
    attacker_id = _card_id(attacker)
    energy_cards = _attached_energy_cards(attacker)
    energy_ids = _attached_energy_card_ids(attacker)
    if attacker_id == HONCHKROW and attack_id == ROCKET_FEATHER_ATTACK:
        return energy_cards <= 0
    if attacker_id == PORYGON2 and attack_id in PORYGON2_R_COMMAND_ATTACKS:
        return energy_cards <= 0 and IGNITION_ENERGY not in energy_ids
    if attacker_id == MURKROW and attack_id in MURKROW_ATTACKS:
        return energy_cards <= 0
    return False


def _planned_energy_available(player, attacker, attack_id):
    attacker_id = _card_id(attacker)
    hand_ids = [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]
    if attacker_id == HONCHKROW and attack_id == ROCKET_FEATHER_ATTACK:
        return TEAM_ROCKET_ENERGY in hand_ids or IGNITION_ENERGY in hand_ids
    if attacker_id == PORYGON2 and attack_id in PORYGON2_R_COMMAND_ATTACKS:
        return IGNITION_ENERGY in hand_ids
    if attacker_id == MURKROW and attack_id in MURKROW_ATTACKS:
        return TEAM_ROCKET_ENERGY in hand_ids
    return False

def _donkarasu_can_ko_from_bench(observation):
    """ベンチのドンカラスがロケットフェザーで相手をKOできるか判定"""
    current, your_index, player = _current_player(observation)
    
    # ベンチからドンカラス取得
    bench = _read(player, "bench", [])
    honchkrow_from_bench = None
    if isinstance(bench, list):
        for slot in bench:
            card = _top_card(slot)
            if _card_id(card) == HONCHKROW:
                honchkrow_from_bench = card
                break
    
    if honchkrow_from_bench is None:
        return False
    
    # ドンカラスがエネルギーを持っているか確認
    energy_cards = _attached_energy_cards(honchkrow_from_bench)
    if energy_cards <= 0:
        return False
    
    # 相手のアクティブHP取得
    opponent_active = _opponent_active_card(observation)
    if opponent_active is None:
        return False
    target_hp = _remaining_hp(opponent_active)
    if target_hp <= 0:
        return False
    
    # ロケットフェザーのダメージ/サポーター計算
    damage_per_supporter = _rocket_feather_damage_per_supporter(opponent_active)
    
    # 相手をKOするために必要なサポーター枚数（切り上げ）
    required_supporters = (target_hp + damage_per_supporter - 1) // damage_per_supporter
    
    # 手札のロケットサポーター枚数
    hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    
    # 必要なサポーター枚数以上あるか
    return hand_supporters >= required_supporters


def _turn_plan_score(player, target, damage, can_ko, prize_count, needs_energy, needs_switch, seed_guard_blocked):
    remaining_prizes = max(1, _prize_cards_remaining(player))
    game_end = can_ko and prize_count >= remaining_prizes
    if game_end:
        score = _policy_rule_number("donkrowTurnPlan", "gameEndScore", 85_000)
    else:
        score = _policy_rule_number("donkrowTurnPlan", "attackBaseScore", 24_000)
    if can_ko:
        score += prize_count * _policy_rule_number("donkrowTurnPlan", "koPrizeScore", 12_500)
    else:
        target_hp = max(1, _remaining_hp(target))
        score += int(min(1.0, max(0, damage) / target_hp) * 5_000)
    score += min(max(0, damage), 360) * _policy_rule_number("donkrowTurnPlan", "damageScore", 18)
    if needs_energy:
        score -= 1_600
    if needs_switch:
        score -= _policy_rule_number("donkrowTurnPlan", "benchPlanPenalty", 4_200)
    if seed_guard_blocked:
        score -= 50_000
    return score, game_end


def _build_donkrow_turn_plan(observation):
    if not _policy_rule_enabled("donkrowTurnPlan"):
        return None
    current, your_index, player = _current_player(observation)
    select = _read(observation, "select", {})
    if _select_context(select) not in (0, "main"):
        return None

    opponent_active = _opponent_active_card(observation)
    if opponent_active is None:
        return None
    target_hp = _remaining_hp(opponent_active)
    if target_hp <= 0:
        return None

    can_switch = _has_switch_option(observation)
    energy_attached = bool(_read(current, "energyAttached", _read(player, "energyAttached", False)))
    seed_guard_blocked = _needs_seed_out_bench_guard(player) and _has_basic_play_option(observation)
    attackers = []
    active = _top_card(_read(player, "active", []))
    if active is not None:
        attackers.append((0, active, False))
    bench = _read(player, "bench", [])
    if isinstance(bench, list):
        for bench_index, slot in enumerate(bench):
            pokemon = _top_card(slot)
            if pokemon is not None:
                attackers.append((bench_index + 1, pokemon, True))

    best_plan = None
    for attacker_index, attacker, needs_switch in attackers:
        if needs_switch and not can_switch:
            continue
        for attack_id in _planned_attack_ids(attacker):
            needs_energy = _planned_attacker_needs_energy(attacker, attack_id)
            if needs_energy and (energy_attached or not _planned_energy_available(player, attacker, attack_id)):
                continue
            damage = _planned_attack_damage(player, attacker, opponent_active, attack_id)
            if damage <= 0:
                continue
            prize_count = _prize_count_for_knockout(opponent_active)
            can_ko = damage >= target_hp
            score, game_end = _turn_plan_score(
                player,
                opponent_active,
                damage,
                can_ko,
                prize_count,
                needs_energy,
                needs_switch,
                seed_guard_blocked,
            )
            attacker_id = _card_id(attacker)
            if attacker_id == HONCHKROW:
                score += _policy_rule_number("donkrowTurnPlan", "honchkrowPlanBonus", 12_000)
            elif attacker_id == MURKROW:
                score -= _policy_rule_number("donkrowTurnPlan", "murkrowAttackPlanPenalty", 28_000)
                if not can_ko:
                    score -= _policy_rule_number("donkrowTurnPlan", "murkrowNonKoPenalty", 18_000)
            elif attacker_id == PORYGON2:
                if not can_ko:
                    score -= max(_policy_rule_number("donkrowTurnPlan", "porygon2NonKoPenalty", 8_000), 28_000)
                if _honchkrow_chain_available(player) and not (can_ko and prize_count >= 2):
                    score -= 18_000
            plan = DonkrowTurnPlan(
                attacker_card=attacker,
                attacker_id=attacker_id,
                attacker_index=attacker_index,
                target_card=opponent_active,
                target_index=0,
                attack_id=attack_id,
                damage=damage,
                remaining_hp=max(0, target_hp - damage),
                prize_count=prize_count,
                score=score,
                can_ko=can_ko,
                needs_energy=needs_energy,
                needs_switch=needs_switch,
                game_end=game_end,
                seed_guard_blocked=seed_guard_blocked,
            )
            if best_plan is None or plan.score > best_plan.score:
                best_plan = plan
    return best_plan


def _turn_plan_attack_score(plan):
    if plan is None:
        return 0
    score = plan.score
    if plan.needs_energy or plan.needs_switch:
        score -= 8_000
    return score if score > 0 else 0


def _is_sakaki_ready_bench_attacker(card):
    identifier = _card_id(card)
    energy_cards = _attached_energy_cards(card)
    energy_ids = _attached_energy_card_ids(card)
    if identifier == HONCHKROW:
        return energy_cards > 0
    if identifier == PORYGON2:
        return energy_cards > 0 or IGNITION_ENERGY in energy_ids
    return False


def _bench_attacker_damage_after_sakaki(player, attacker, target, giovanni_from_hand):
    if not _is_sakaki_ready_bench_attacker(attacker):
        return 0
    identifier = _card_id(attacker)
    if identifier == HONCHKROW:
        hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)
        fuel_after_sakaki = max(0, hand_supporters - (1 if giovanni_from_hand else 0))
        if fuel_after_sakaki <= 0:
            return 0
        return _rocket_feather_damage_per_supporter(target) * min(4, fuel_after_sakaki)
    if identifier == PORYGON2:
        discard_supporters = _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS)
        return (discard_supporters + 1) * _policy_rule_number("porygon2RCommandFallback", "damagePerTrashRocketSupporter", 20)
    return 0


def _sakaki_two_prize_ko_score(observation, giovanni_from_hand=True):
    if not _policy_rule_enabled("sakakiRequiresKo"):
        return None
    current, your_index, player = _current_player(observation)
    active = _top_card(_read(player, "active", []))
    if _card_id(active) not in ROCKET_FIELD_POKEMON:
        return None

    min_prize = _policy_rule_number("sakakiRequiresKo", "minPrizeScore", 2)
    bench_attackers = [
        pokemon
        for slot in _read(player, "bench", [])
        if (pokemon := _top_card(slot)) is not None and _is_sakaki_ready_bench_attacker(pokemon)
    ]
    if not bench_attackers:
        return None

    targets = []
    for target in _opponent_bench_cards(observation):
        remaining_hp = _remaining_hp(target)
        prize_count = _prize_count_for_knockout(target)
        if remaining_hp > 0 and prize_count >= min_prize:
            targets.append((target, remaining_hp, prize_count))
    if not targets:
        return None

    best_score = None
    for attacker in bench_attackers:
        for target, remaining_hp, prize_count in targets:
            damage = _bench_attacker_damage_after_sakaki(player, attacker, target, giovanni_from_hand)
            if damage < remaining_hp:
                continue
            score = (
                _policy_rule_number("sakakiRequiresKo", "koBaseScore", 95_000) +
                prize_count * _policy_rule_number("sakakiRequiresKo", "koPerPrizeScore", 75_000) +
                min(damage, 360) * 12
            )
            if _card_id(attacker) == HONCHKROW:
                score += 4_500
            if prize_count >= 3:
                score += 20_000
            if best_score is None or score > best_score:
                best_score = score
    return best_score


def _rocket_attack_needs_energy(player):
    active = _top_card(_read(player, "active", []))
    active_id = _card_id(active)
    if active_id in (HONCHKROW, PORYGON2, MURKROW) and _attached_energy_cards(active) <= 0:
        return True
    return any(
        _card_id(card) in (HONCHKROW, PORYGON2, MURKROW)
        and _attached_energy_cards(card) <= 0
        for card in _field_top_cards(player)
    )


def _rocket_feather_fuel_needed(player):
    field_ids = _field_card_ids(player)
    hand_ids = [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]
    future_honchkrow_line = (
        HONCHKROW in field_ids
        or (MURKROW in field_ids and (HONCHKROW in hand_ids or _count_cards(player, ("deck",), lambda card_id: card_id == HONCHKROW) > 0))
    )
    hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    return future_honchkrow_line and hand_supporters < 3


def _should_avoid_apollo_reset(observation, apollo_from_hand=True):
    current, _, player = _current_player(observation)
    hand_count = _hand_count(player)
    deck_count = _deck_count(player)
    remaining_deck_after_reset = deck_count + max(0, hand_count - (1 if apollo_from_hand else 0))
    if remaining_deck_after_reset <= _policy_rule_number("rocketApolloReset", "lowDeckResetThreshold", 6):
        return True

    hand_energy = _count_cards(player, ("hand",), lambda card_id: card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY))
    hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    hand_supporters_after_apollo = max(0, hand_supporters - (1 if apollo_from_hand else 0))
    discard_supporters = _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    hand_ids = [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]
    needs_energy = _rocket_attack_needs_energy(player)
    needs_fuel = _rocket_feather_fuel_needed(player)
    has_athena_alternative = ARIANA in hand_ids and not _supporter_played_this_turn(current, player)
    has_headset_alternative = MIRACLE_HEADSET in hand_ids

    if hand_supporters_after_apollo >= 3 and not needs_energy:
        return True
    if hand_energy > 0 and needs_energy:
        return True
    if needs_fuel and has_headset_alternative and discard_supporters >= 2:
        return True
    if has_athena_alternative and (needs_energy or needs_fuel):
        return True
    return False


def _apollo_reset_score(observation, assume_legal=False, apollo_from_hand=True):
    if not _policy_rule_enabled("rocketApolloReset"):
        return _policy_rule_number("rocketApolloReset", "unavailableScore", -9_500)
    current, _, player = _current_player(observation)
    if _supporter_played_this_turn(current, player):
        return -7_500

    lost_signal = _lost_rocket_mon_last_turn(observation, current, player)
    if lost_signal is False or (lost_signal is None and not assume_legal):
        return _policy_rule_number("rocketApolloReset", "unavailableScore", -9_500)
    if _should_avoid_apollo_reset(observation, apollo_from_hand=apollo_from_hand):
        return _policy_rule_number("rocketApolloReset", "avoidScore", -8_500)

    field_ids = _field_card_ids(player)
    hand_count = _hand_count(player)
    hand_energy = _count_cards(player, ("hand",), lambda card_id: card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY))
    deck_energy = _count_cards(player, ("deck",), lambda card_id: card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY))
    deck_evolutions = _count_cards(player, ("deck",), lambda card_id: card_id in (HONCHKROW, PORYGON2))
    needs_energy = _rocket_attack_needs_energy(player)
    needs_fuel = _rocket_feather_fuel_needed(player)
    has_murkrow_line = MURKROW in field_ids or HONCHKROW in field_ids

    if hand_count <= 3:
        hand_pressure = _policy_rule_number("rocketApolloReset", "shortHandScore", 8_000)
    elif hand_count <= 5:
        hand_pressure = _policy_rule_number("rocketApolloReset", "mediumHandScore", 4_000)
    else:
        hand_pressure = _policy_rule_number("rocketApolloReset", "largeHandScore", 1_200)

    score = _policy_rule_number("rocketApolloReset", "baseScore", 7_500) + hand_pressure
    score += _policy_rule_number("rocketApolloReset", "recoveryBaseScore", 18_000)
    if has_murkrow_line:
        score += _policy_rule_number("rocketApolloReset", "murkrowLineBonus", 4_500)
    if hand_energy == 0 and deck_energy > 0:
        score += _policy_rule_number("rocketApolloReset", "noHandEnergyBonus", 8_500)
    if needs_energy:
        score += _policy_rule_number("rocketApolloReset", "needsEnergyBonus", 10_000)
    if needs_fuel:
        score += _policy_rule_number("rocketApolloReset", "needsFuelBonus", 6_500)
    score += min(3, deck_evolutions) * _policy_rule_number("rocketApolloReset", "perEvolutionScore", 2_400)
    score += min(5, _opponent_hand_count(observation)) * _policy_rule_number("rocketApolloReset", "opponentHandPressureScore", 500)
    return score


def _apollo_search_score(observation):
    score = _apollo_reset_score(observation, assume_legal=False, apollo_from_hand=False)
    if score > 0:
        return score - _policy_rule_number("rocketApolloReset", "searchPenalty", 2_200)
    _, _, player = _current_player(observation)
    if _rocket_feather_fuel_needed(player):
        return _policy_rule_number("rocketApolloReset", "fuelOnlyScore", 900)
    return score


def _opponent_active_attack_threat(observation):
    opponent_active = _opponent_active_card(observation)
    identifier = _card_id(opponent_active)
    if identifier is None:
        return 0
    energy_cards = _attached_energy_cards(opponent_active)
    energy_units = _attached_energy_units(opponent_active)
    energy_ids = _attached_energy_card_ids(opponent_active)
    has_special_energy = energy_cards > 0 or bool(energy_ids)
    if identifier == HONCHKROW:
        return 180 if has_special_energy or energy_units >= 2 else 0
    if identifier == PORYGON2:
        return 120 if has_special_energy or energy_units >= 1 else 0
    if identifier == MURKROW:
        return 30 if energy_units >= 1 or has_special_energy else 0
    return 80 if energy_units >= 2 or has_special_energy else 0


def _has_main_attack_option(observation, attack_id):
    select = _read(observation, "select", {})
    for option in _select_options(select):
        if _option_type(option) in (13, "attack") and _attack_id(option) == attack_id:
            return True
    return False


def _has_switch_option(observation):
    select = _read(observation, "select", {})
    for option in _select_options(select):
        if _option_type(option) in (12, "retreat", "switch", "promote"):
            return True
    return False


def _has_any_main_attack_option(observation):
    select = _read(observation, "select", {})
    return any(_option_type(option) in (13, "attack") for option in _select_options(select))


def _ready_bench_attacker_ids(player):
    ready = []
    bench = _read(player, "bench", [])
    if not isinstance(bench, list):
        return ready
    for slot in bench:
        pokemon = _top_card(slot)
        identifier = _card_id(pokemon)
        energy_cards = _attached_energy_cards(pokemon)
        energy_ids = _attached_energy_card_ids(pokemon)
        if identifier == HONCHKROW and energy_cards > 0:
            ready.append(identifier)
        elif identifier == PORYGON2 and (energy_cards > 0 or IGNITION_ENERGY in energy_ids):
            ready.append(identifier)
    return ready


def _has_ariana_compression_option(observation):
    select = _read(observation, "select", {})
    for option in _select_options(select):
        if _is_ariana_compression_option(observation, option):
            return True
    return False


def _all_field_pokemon_are_rocket(player):
    field_ids = _field_card_ids(player)
    return bool(field_ids) and all(card_id in ROCKET_FIELD_POKEMON for card_id in field_ids)


def _ariana_draw_count_for_player(player):
    hand_count = _hand_count(player)
    target_hand = 8 if _all_field_pokemon_are_rocket(player) else 5
    return max(0, target_hand - max(0, hand_count - 1))


def _needs_ariana_energy_dig(player):
    hand_energy = _count_cards(player, ("hand",), lambda card_id: card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY))
    deck_energy = _count_cards(player, ("deck",), lambda card_id: card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY))
    if hand_energy > 0 or deck_energy <= 0:
        return False
    active = _top_card(_read(player, "active", []))
    active_id = _card_id(active)
    active_needs_energy = active_id in (HONCHKROW, PORYGON2, MURKROW) and _attached_energy_cards(active) <= 0
    field_ids = _field_card_ids(player)
    has_attack_line = any(card_id in (HONCHKROW, PORYGON2, MURKROW, PORYGON) for card_id in field_ids)
    return active_needs_energy or has_attack_line


def _donkrow_development_settled(player):
    field_top_ids = [_card_id(card) for card in _field_top_cards(player)]
    hand_ids = [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]
    murkrow_lines = field_top_ids.count(MURKROW) + field_top_ids.count(HONCHKROW)
    has_honchkrow_access = HONCHKROW in field_top_ids or HONCHKROW in hand_ids
    return murkrow_lines >= 2 and has_honchkrow_access


def _only_porygon_board(player):
    field_top_ids = [_card_id(card) for card in _field_top_cards(player)]
    return (
        bool(field_top_ids)
        and not any(card_id in (MURKROW, HONCHKROW) for card_id in field_top_ids)
        and any(card_id in (PORYGON, PORYGON2) for card_id in field_top_ids)
    )


def _porygon_development_allowed(player):
    return _donkrow_development_settled(player) or _only_porygon_board(player)


def _factory_in_hand(player):
    return any(_card_id(card) == FACTORY for card in _iter_cards(_read(player, "hand", [])))


def _factory_option_available(observation):
    select = _read(observation, "select", {})
    if _select_context(select) not in (0, "main"):
        return False
    for option in _select_options(select):
        if _card_id(_card_from_option(observation, option)) == FACTORY and _option_type(option) in (7, 10, "play", "ability"):
            return True
    return False


def _honchkrow_chain_available(player, hand_ids=None, deck_ids=None):
    if hand_ids is None:
        hand_ids = [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]
    if deck_ids is None:
        deck_ids = [_card_id(card) for card in _iter_cards(_read(player, "deck", []))]
    has_energy = TEAM_ROCKET_ENERGY in hand_ids or IGNITION_ENERGY in hand_ids
    has_honchkrow_access = HONCHKROW in hand_ids or HONCHKROW in deck_ids
    for card in _field_top_cards(player):
        identifier = _card_id(card)
        if identifier == HONCHKROW:
            if _attached_energy_cards(card) > 0 or has_energy:
                return True
        if identifier == MURKROW and has_honchkrow_access and has_energy:
            return True
    return False


def _bench_honchkrow_promotion_available(player, hand_ids=None, deck_ids=None):
    if hand_ids is None:
        hand_ids = [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]
    if deck_ids is None:
        deck_ids = [_card_id(card) for card in _iter_cards(_read(player, "deck", []))]
    has_energy = TEAM_ROCKET_ENERGY in hand_ids or IGNITION_ENERGY in hand_ids
    has_honchkrow_access = HONCHKROW in hand_ids or HONCHKROW in deck_ids
    bench = _read(player, "bench", [])
    if not isinstance(bench, list):
        return False
    for slot in bench:
        pokemon = _top_card(slot)
        identifier = _card_id(pokemon)
        if identifier == HONCHKROW and (_attached_energy_cards(pokemon) > 0 or has_energy):
            return True
        if identifier == MURKROW and has_honchkrow_access and has_energy:
            return True
    return False


def _honchkrow_chain_promotion_score(player, promotion_id, promotion_card, hand_ids, deck_ids, hand_supporters):
    has_energy = TEAM_ROCKET_ENERGY in hand_ids or IGNITION_ENERGY in hand_ids
    has_rocket_energy = TEAM_ROCKET_ENERGY in hand_ids
    has_honchkrow_in_hand = HONCHKROW in hand_ids
    has_honchkrow_access = has_honchkrow_in_hand or HONCHKROW in deck_ids
    has_fuel = hand_supporters > 0
    if promotion_id == HONCHKROW:
        if _attached_energy_cards(promotion_card) > 0 and has_fuel:
            return 76_000
        if has_energy and has_fuel:
            return 70_000 + (6_000 if has_rocket_energy else 0)
        return 34_000
    if promotion_id == MURKROW:
        if has_honchkrow_in_hand and has_energy and has_fuel:
            return 82_000 + (6_000 if has_rocket_energy else 0)
        if has_honchkrow_access and has_energy:
            return 46_000
        if has_honchkrow_access:
            return 28_000
    if promotion_id in (PORYGON, PORYGON2) and _bench_honchkrow_promotion_available(player, hand_ids, deck_ids):
        return -36_000 if promotion_id == PORYGON else -22_000
    return None


def _poke_pad_evolution_attack_need(player, hand_ids):
    field_top_ids = [_card_id(card) for card in _field_top_cards(player)]
    active_id = _card_id(_top_card(_read(player, "active", [])))
    has_attack_energy = TEAM_ROCKET_ENERGY in hand_ids or IGNITION_ENERGY in hand_ids
    needs_honchkrow = MURKROW in field_top_ids and HONCHKROW not in hand_ids and (
        has_attack_energy or any(_card_id(card) == MURKROW and _attached_energy_cards(card) > 0 for card in _field_top_cards(player))
    )
    needs_porygon2 = (
        _porygon_development_allowed(player)
        and PORYGON in field_top_ids
        and PORYGON2 not in hand_ids
        and IGNITION_ENERGY in hand_ids
    )
    active_bonus = (active_id == MURKROW and needs_honchkrow) or (active_id == PORYGON and needs_porygon2)
    return needs_honchkrow, needs_porygon2, active_bonus


def _supporter_played_this_turn(current, player):
    return bool(
        _read(player, "supporterPlayed", False)
        or _read(player, "supporterUsedThisTurn", False)
        or _read(current, "supporterPlayed", False)
        or _read(current, "supporterUsedThisTurn", False)
        or _read(current, "rocketSupporterUsedThisTurn", False)
    )


def _rocket_supporter_history_count(player):
    return _count_cards(player, ("discard", "trash"), lambda card_id: card_id in ROCKET_SUPPORTERS)


def _proton_opening_allowed(current, player, setup_incomplete):
    return (
        bool(setup_incomplete)
        and not _supporter_played_this_turn(current, player)
        and _rocket_supporter_history_count(player) <= 0
    )


def _turn_number(current):
    return _safe_int(_read(current, "turn", _read(current, "turnNumber", _read(current, "turn_number", 0))), 0)


def _turn_one_proton_priority(current, player):
    return _turn_number(current) <= 1 and not _supporter_played_this_turn(current, player)


def _stadium_played_this_turn(current, player):
    return bool(
        _read(player, "stadiumPlayed", False)
        or _read(player, "stadiumUsedThisTurn", False)
        or _read(current, "stadiumPlayed", False)
        or _read(current, "stadiumUsedThisTurn", False)
    )


def _is_initial_deck_request(observation, select):
    return select is None


def _choose_rocket_feather_costs(observation, options, max_count):
    select = _read(observation, "select", {})
    effect = _read(select, "effect")
    context = _select_context(select)
    if _card_id(effect) != HONCHKROW and context != 8:
        return None

    current = _read(observation, "current", {})
    your_index = _safe_int(_read(current, "yourIndex"), 0)
    opponent = _player_state(current, 1 - your_index)
    opponent_active = _top_card(_read(opponent, "active", []))
    parsed_remaining_hp = _remaining_hp(opponent_active)
    damage_per_supporter = _rocket_feather_damage_per_supporter(opponent_active)
    required_for_ko = max(1, (parsed_remaining_hp + damage_per_supporter - 1) // damage_per_supporter) if parsed_remaining_hp > 0 else 1
    can_ko = parsed_remaining_hp > 0 and required_for_ko <= max_count
    required = min(max_count, required_for_ko)
    if not can_ko:
        required = min(max_count, 2 if max_count >= 3 else 1)

    candidates = []
    for option_index, option in enumerate(options):
        card = _card_from_option(observation, option)
        identifier = _card_id(card)
        if identifier not in ROCKET_SUPPORTER_COST_PRIORITY:
            continue
        candidates.append((ROCKET_SUPPORTER_COST_PRIORITY[identifier], option_index, identifier))

    if not candidates:
        return []

    protected = {ARIANA, PETREL}
    expendable = [item for item in candidates if item[2] not in protected]
    if len(expendable) >= required:
        candidates = expendable
    else:
        non_ariana = [item for item in candidates if item[2] != ARIANA]
        if len(non_ariana) >= required:
            candidates = non_ariana

    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = [option_index for _, option_index, _ in candidates[:required]]
    return selected[:max_count]


def _move_lock_option_name(option):
    for key in ("moveName", "move_name", "name", "label", "text", "title"):
        value = _read(option, key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    move = _read(option, "move")
    if isinstance(move, dict):
        return _move_lock_option_name(move)
    return ""


def _move_lock_option_damage(option):
    for key in ("damage", "baseDamage", "base_damage", "power", "amount"):
        value = _read(option, key)
        if value is not None:
            return _safe_int(value, 0)
    move = _read(option, "move")
    if isinstance(move, dict):
        return _move_lock_option_damage(move)
    return 0


def _choose_taunt_move_lock_option(observation, options, max_count):
    if max_count != 1 or not options:
        return None
    if any(_card_from_option(observation, option) is not None for option in options):
        return None
    named_options = []
    for option_index, option in enumerate(options):
        name = _move_lock_option_name(option)
        if name:
            named_options.append((option_index, name, _move_lock_option_damage(option)))
    if not named_options:
        return None
    select_text = str(_select_payload(observation)).lower()
    if not any(marker in select_text for marker in ("taunt", "いちゃもん", "move_lock", "move lock", "lock")):
        return None
    best = max(named_options, key=lambda item: (item[2], item[1]))
    return [best[0]]


def _field_counts(current, player_index):
    player = _player_state(current, player_index)
    counts = {}
    for zone in ("active", "bench", "hand", "discard", "trash"):
        for card in _iter_cards(_read(player, zone, [])):
            identifier = _card_id(card)
            counts[identifier] = counts.get(identifier, 0) + 1
    return counts


def _rank_optional_card(observation, option):
    current = _read(observation, "current", {})
    select = _read(observation, "select", {})
    your_index = _safe_int(_read(current, "yourIndex"), 0)
    context = _select_context(select)
    option_type = _option_type(option)
    card = _card_from_option(observation, option)
    identifier = _card_id(card)
    counts = _field_counts(current, your_index)
    player = _player_state(current, your_index)

    if identifier is None or option_type not in (3, "card"):
        return -10_000

    if context in (2, "setup bench pokemon", "setup bench"):
        if identifier == MURKROW:
            return 1_000 - counts.get(MURKROW, 0) * 40
        if identifier == PORYGON:
            return 760 - counts.get(PORYGON, 0) * 80
        return -1_000

    if identifier in BASIC_SETUP_POKEMON:
        if identifier == MURKROW:
            return 950 - (counts.get(MURKROW, 0) + counts.get(HONCHKROW, 0)) * 70
        return 760 - counts.get(PORYGON, 0) * 120
    if identifier == PORYGON2:
        if not _porygon_development_allowed(player):
            return 120
        return 900 if counts.get(PORYGON, 0) > 0 else 180
    if identifier == HONCHKROW:
        return 680 if counts.get(MURKROW, 0) > 0 else 240
    if identifier == TEAM_ROCKET_ENERGY:
        return 420
    if identifier == IGNITION_ENERGY:
        return 390
    if identifier == PROTON:
        setup_incomplete = _bench_top_count(player) < 2 or counts.get(MURKROW, 0) + counts.get(HONCHKROW, 0) < 2
        return 900 if _proton_opening_allowed(current, player, setup_incomplete) else -5_000
    if identifier == PETREL:
        return 570
    if identifier == ARIANA:
        return 520
    if identifier == GIOVANNI:
        return -5_000
    if identifier == ARCHER:
        return 430
    if identifier == TEAM_ROCKET_TRANSCEIVER:
        return 610
    if identifier == POKE_PAD:
        return 540
    if identifier == POKEGEAR:
        return 500
    if identifier == ROTO_STICK:
        return 480
    if identifier == NIGHT_STRETCHER:
        return 340
    if identifier == MIRACLE_HEADSET:
        return 260
    if identifier == FACTORY:
        if _supporter_played_this_turn(current, player) and not _stadium_played_this_turn(current, player):
            return 2_400
        return 300
    return 0


def _choose_optional_cards(observation, options, max_count):
    current = _read(observation, "current", {})
    select = _read(observation, "select", {})
    your_index = _safe_int(_read(current, "yourIndex"), 0)
    counts = _field_counts(current, your_index)
    effect = _effect_id(observation)
    context = _select_context(select)
    _, _, player = _current_player(observation)
    field_ids = _field_card_ids(player)
    field_top_ids = [_card_id(field_card) for field_card in _field_top_cards(player)]
    hand_ids = [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]
    deck_ids = [_card_id(card) for card in _iter_cards(_read(player, "deck", []))]
    active_ids = [_card_id(card) for card in _iter_cards(_read(player, "active", []))]
    active_card = _top_card(_read(player, "active", []))
    active_honchkrow = any(card_id == HONCHKROW for card_id in active_ids)
    has_rocket_feather_action = _has_main_attack_option(observation, ROCKET_FEATHER_ATTACK)
    hand_count = _hand_count(player)
    hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    discard_supporters = _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    turn_number = _safe_int(_read(current, "turn"), 0)
    all_rocket_field = _all_field_pokemon_are_rocket(player)
    athena_draw_count = _ariana_draw_count_for_player(player)
    murkrow_lines = field_ids.count(MURKROW) + field_ids.count(HONCHKROW)
    porygon_lines = field_ids.count(PORYGON) + field_ids.count(PORYGON2)
    bench_count = _zone_count(player, "bench")
    setup_incomplete = bench_count < 2 or murkrow_lines < 2 or porygon_lines < 1
    proton_opening_allowed = _proton_opening_allowed(current, player, setup_incomplete)
    turn_one_proton_priority = _turn_one_proton_priority(current, player)
    early_turn = turn_number <= _policy_rule_number("preferProtonWhenSetupIncomplete", "earlyTurnMax", 4)
    supporter_played = _supporter_played_this_turn(current, player)
    stadium_ids = [_card_id(card) for card in _iter_cards(_read(current, "stadium", []))]
    search_option_ids = [_card_id(_card_from_option(observation, option)) for option in options]
    search_can_take_proton = PROTON in search_option_ids
    energy_dig_needed = _needs_ariana_energy_dig(player)
    needs_honchkrow_from_pad, needs_porygon2_from_pad, active_pad_evolution = _poke_pad_evolution_attack_need(player, hand_ids)
    porygon_development_allowed = _porygon_development_allowed(player)
    hand_has_compression = any(
        card_id in (POKEGEAR, POKE_PAD, ROTO_STICK, TEAM_ROCKET_TRANSCEIVER, MURKROW, PORYGON, HONCHKROW, PORYGON2)
        for card_id in hand_ids
    )
    energy_murkrow_needs_honchkrow = _energy_prepared_murkrow_needs_honchkrow(player, hand_ids, deck_ids)
    petrel_attack_bridge = energy_murkrow_needs_honchkrow or needs_honchkrow_from_pad or needs_porygon2_from_pad or active_pad_evolution
    active_attack_ready = active_honchkrow and has_rocket_feather_action and hand_supporters > 0
    future_honchkrow_line = (
        active_honchkrow
        or HONCHKROW in field_top_ids
        or (MURKROW in field_top_ids and (HONCHKROW in hand_ids or HONCHKROW in deck_ids))
    )
    needs_rocket_feather_fuel = future_honchkrow_line and hand_supporters < 3

    def giovanni_fuel_search_score():
        sakaki_ko_score = _sakaki_two_prize_ko_score(observation, giovanni_from_hand=False)
        if not supporter_played and sakaki_ko_score is not None:
            return sakaki_ko_score - _policy_rule_number("sakakiRequiresKo", "searchPenalty", 3_200)
        if energy_dig_needed and ARIANA in search_option_ids:
            return -100_000
        if not needs_rocket_feather_fuel:
            return -100_000
        missing_supporters = max(0, 3 - hand_supporters)
        score = 900 + missing_supporters * 850
        if active_honchkrow or has_rocket_feather_action:
            score += 1_200
        if TEAM_ROCKET_ENERGY in hand_ids or IGNITION_ENERGY in hand_ids:
            score += 650
        return score

    selected = []
    available = list(enumerate(options))
    while available and len(selected) < max_count:
        best = None
        for option_index, option in available:
            score = _rank_optional_card(observation, option)
            option_card = _card_from_option(observation, option)
            identifier = _card_id(option_card)
            if effect == PROTON and context in (7, "to hand search"):
                murkrow_lines = counts.get(MURKROW, 0) + counts.get(HONCHKROW, 0)
                porygon_lines = counts.get(PORYGON, 0) + counts.get(PORYGON2, 0)
                if identifier == MURKROW:
                    score = 2_000 if murkrow_lines < 3 else 250
                elif identifier == PORYGON:
                    score = 1_700 if porygon_lines < 1 else 220
                else:
                    score = -10_000
            elif turn_one_proton_priority and identifier == PROTON and PROTON not in hand_ids and effect in (POKEGEAR, TEAM_ROCKET_TRANSCEIVER, ROTO_STICK):
                score = 250_000
            elif effect == POKE_PAD:
                field_murkrow_lines = field_top_ids.count(MURKROW) + field_top_ids.count(HONCHKROW)
                field_porygon_lines = field_top_ids.count(PORYGON) + field_top_ids.count(PORYGON2)
                desired_murkrow_lines = 3 if turn_number <= 2 else 2
                has_honchkrow_in_hand = HONCHKROW in hand_ids
                has_porygon_attacker_in_hand = PORYGON2 in hand_ids
                seed_guard_needs_basic = _needs_seed_out_bench_guard(player) and not _basic_setup_ids_in_hand(player)
                seed_guard_has_basic_option = any(
                    _is_basic_setup_pokemon(_card_id(_card_from_option(observation, available_option)))
                    for _, available_option in available
                )
                if seed_guard_needs_basic and seed_guard_has_basic_option and not _is_basic_setup_pokemon(identifier):
                    score = -100_000
                elif identifier == MURKROW:
                    score = max(_poke_pad_target_score(observation, option_card), 2_100 if field_murkrow_lines < desired_murkrow_lines else 260)
                    if active_attack_ready and field_murkrow_lines < 3:
                        score = max(score, 40_000)
                elif identifier == HONCHKROW:
                    score = max(
                        _poke_pad_target_score(observation, option_card),
                        4_900 if energy_murkrow_needs_honchkrow else (2_300 if field_murkrow_lines > 0 and not has_honchkrow_in_hand else 420),
                    )
                    if needs_honchkrow_from_pad:
                        score += _policy_rule_number("pokePadEvolutionAttack", "honchkrowWithEnergyInHandScore", 12_000)
                    if active_pad_evolution and identifier == HONCHKROW:
                        score += _policy_rule_number("pokePadEvolutionAttack", "activeEvolutionBonus", 2_800)
                elif identifier == PORYGON:
                    score = max(_poke_pad_target_score(observation, option_card), 1_650 if field_porygon_lines < 1 else 220)
                elif identifier == PORYGON2:
                    if not porygon_development_allowed:
                        score = 260
                    elif _honchkrow_chain_available(player, hand_ids, deck_ids) and not _only_porygon_board(player) and not active_pad_evolution:
                        score = 520
                    else:
                        score = max(
                            _poke_pad_target_score(observation, option_card),
                            2_850 if field_porygon_lines > 0 and not has_porygon_attacker_in_hand else 420,
                        )
                        if needs_porygon2_from_pad:
                            score += _policy_rule_number("pokePadEvolutionAttack", "porygon2WithIgnitionInHandScore", 11_800)
                        if active_pad_evolution and identifier == PORYGON2:
                            score += _policy_rule_number("pokePadEvolutionAttack", "activeEvolutionBonus", 2_800)
                else:
                    score = -10_000
            elif effect == POKEGEAR:
                if identifier == PROTON:
                    if PROTON in hand_ids:
                        score -= _policy_rule_number("preferProtonWhenSetupIncomplete", "settledSearchPenalty", 50_000)
                    elif proton_opening_allowed:
                        score += 12_000
                    else:
                        score -= _policy_rule_number("preferProtonWhenSetupIncomplete", "settledSearchPenalty", 50_000)
                    if energy_murkrow_needs_honchkrow:
                        score -= 2_700
                elif identifier == PETREL:
                    if petrel_attack_bridge:
                        score += 2_900
                    elif ARIANA not in hand_ids and FACTORY not in stadium_ids:
                        score += 1_500
                    else:
                        score -= 1_200
                elif identifier == ARIANA:
                    score += 12_000 if not proton_opening_allowed else 0
                    score += 1_250 if athena_draw_count >= 3 or hand_count <= 5 else 520
                    if energy_dig_needed:
                        score += _policy_rule_number("preferArianaEnergyDig", "pokegearArianaBonus", 5_200)
                    if _policy_rule_enabled("compressBeforeAriana") and hand_count >= 6 and hand_has_compression:
                        score -= _policy_rule_number("preferProtonWhenSetupIncomplete", "arianaPokegearCompressionPenalty", 1_800)
                elif identifier == ARCHER:
                    score = _apollo_search_score(observation)
                elif identifier == GIOVANNI:
                    score = giovanni_fuel_search_score()
            elif effect == MIRACLE_HEADSET:
                if identifier == ARIANA:
                    score = 42_000 + (6_000 if energy_dig_needed or athena_draw_count >= 3 else 0)
                elif identifier == ARCHER:
                    score = max(_policy_rule_number("rocketApolloReset", "fuelOnlyScore", 900), _apollo_search_score(observation))
                elif identifier == PETREL:
                    score = 24_000
                elif identifier == GIOVANNI:
                    score = max(14_000, giovanni_fuel_search_score())
                elif identifier == PROTON:
                    score = 2_000 if proton_opening_allowed else _policy_rule_number("preferProtonWhenSetupIncomplete", "settledRecoveryScore", -50_000)
            elif effect == TEAM_ROCKET_TRANSCEIVER:
                has_proton_in_hand = PROTON in hand_ids
                has_ariana_in_hand = ARIANA in hand_ids
                if has_ariana_in_hand:
                    if identifier == ARIANA:
                        score = -8_000
                    elif identifier == PETREL:
                        score = 18_000 + (5_200 if petrel_attack_bridge else 0)
                    elif identifier == ARCHER:
                        score = max(18_000, _apollo_search_score(observation))
                    elif identifier == GIOVANNI:
                        score = max(11_000, giovanni_fuel_search_score())
                    elif identifier == PROTON:
                        score = -_policy_rule_number("preferProtonWhenSetupIncomplete", "settledSearchPenalty", 50_000)
                elif has_proton_in_hand:
                    if identifier == ARIANA:
                        score = 40_000 + (6_000 if energy_dig_needed or athena_draw_count >= 3 else 0)
                    elif identifier == PETREL:
                        score = 12_000 + (5_200 if petrel_attack_bridge else 0)
                    elif identifier == ARCHER:
                        score = _apollo_search_score(observation)
                    elif identifier == GIOVANNI:
                        score = max(9_200, giovanni_fuel_search_score())
                    elif identifier == PROTON:
                        score = -_policy_rule_number("preferProtonWhenSetupIncomplete", "settledSearchPenalty", 50_000)
                elif identifier == PROTON and not supporter_played:
                    score = 30_000 if proton_opening_allowed else -_policy_rule_number("preferProtonWhenSetupIncomplete", "settledSearchPenalty", 50_000)
                elif identifier == ARIANA:
                    score += 12_000 if not proton_opening_allowed else 1_450
                    score += 1_450 if athena_draw_count >= 3 or hand_count <= 5 else 520
                    if energy_dig_needed:
                        score += _policy_rule_number("preferArianaEnergyDig", "transceiverArianaBonus", 8_800)
                    if proton_opening_allowed and search_can_take_proton and (murkrow_lines < 2 or porygon_lines < 1) and not energy_dig_needed:
                        score -= _policy_rule_number("preferProtonWhenSetupIncomplete", "arianaOverProtonPenalty", 2_400)
                    if _policy_rule_enabled("compressBeforeAriana") and hand_count >= 6 and hand_has_compression:
                        score -= _policy_rule_number("preferProtonWhenSetupIncomplete", "arianaSearchCompressionPenalty", 1_400)
                elif identifier == PROTON:
                    if proton_opening_allowed:
                        score += _policy_rule_number("preferProtonWhenSetupIncomplete", "transceiverSetupBonus", 3_200)
                    else:
                        score -= _policy_rule_number("preferProtonWhenSetupIncomplete", "settledSearchPenalty", 50_000)
                    if energy_murkrow_needs_honchkrow:
                        score -= 3_400
                elif identifier == PETREL:
                    if petrel_attack_bridge:
                        score += 5_600
                    elif ARIANA not in hand_ids and FACTORY not in stadium_ids:
                        score += 1_800
                    else:
                        score -= 1_400
                elif identifier == GIOVANNI:
                    score = giovanni_fuel_search_score()
                elif identifier == ARCHER:
                    score = _apollo_search_score(observation)
            elif effect == PETREL:
                murkrow_lines = field_ids.count(MURKROW) + field_ids.count(HONCHKROW)
                porygon_lines = field_ids.count(PORYGON) + field_ids.count(PORYGON2)
                needs_setup = murkrow_lines < 2 or porygon_lines < 1
                petrel_should_chain_factory = ARIANA not in hand_ids and FACTORY not in stadium_ids
                if identifier == POKE_PAD:
                    score += 7_200 if petrel_attack_bridge else (1_100 if needs_setup else 520)
                elif identifier == TEAM_ROCKET_TRANSCEIVER:
                    score += 2_800 if petrel_attack_bridge else 950
                elif identifier == POKEGEAR:
                    score += 2_200 if petrel_attack_bridge else 720
                elif identifier == MIRACLE_HEADSET:
                    score += 2_900 if active_honchkrow and hand_supporters < 3 and discard_supporters > 0 else -900
                elif identifier == FACTORY:
                    if _factory_in_hand(player) or FACTORY in stadium_ids:
                        score -= 100_000
                    else:
                        score += 52_000 if petrel_should_chain_factory else (18_000 if supporter_played and FACTORY not in stadium_ids else 520)
                    if not supporter_played and energy_dig_needed and ARIANA in hand_ids:
                        score -= _policy_rule_number("preferArianaEnergyDig", "factoryBeforeArianaPenalty", 6_000)
                elif identifier == ARIANA:
                    if petrel_should_chain_factory:
                        score -= 2_400
                    score += 1_150 if athena_draw_count >= 3 or hand_count <= 5 else 340
                    if energy_dig_needed:
                        score += _policy_rule_number("preferArianaEnergyDig", "petrelArianaBonus", 4_600)
                elif identifier == PETREL:
                    score -= 450
                elif identifier == GIOVANNI:
                    score = giovanni_fuel_search_score()
                elif identifier == ARCHER:
                    score = _apollo_search_score(observation)
                elif identifier == NIGHT_STRETCHER:
                    score += max(0, _best_night_stretcher_target_score(observation) // 60)
            elif effect == MURKROW:
                if identifier not in ROCKET_SUPPORTERS:
                    score = -100_000
                elif identifier == PROTON:
                    if PROTON in hand_ids:
                        score = -_policy_rule_number("preferProtonWhenSetupIncomplete", "settledSearchPenalty", 50_000)
                    elif proton_opening_allowed:
                        score = 30_000
                    else:
                        score = -_policy_rule_number("preferProtonWhenSetupIncomplete", "settledSearchPenalty", 50_000)
                    if energy_murkrow_needs_honchkrow:
                        score -= 3_400
                elif identifier == ARIANA:
                    score = 36_000 + (6_000 if energy_dig_needed or athena_draw_count >= 3 or hand_count <= 5 else 0)
                    if energy_dig_needed:
                        score += _policy_rule_number("preferArianaEnergyDig", "transceiverArianaBonus", 8_800)
                elif identifier == PETREL:
                    score = 18_000 + (2_200 if energy_murkrow_needs_honchkrow else 0)
                elif identifier == ARCHER:
                    score = _apollo_search_score(observation)
                elif identifier == GIOVANNI:
                    score = max(9_200, giovanni_fuel_search_score())
            elif effect == ROTO_STICK:
                if identifier in SUPPORTER_CARD_IDS:
                    score += 40_000
                elif identifier == MIRACLE_HEADSET:
                    score += 2_800 if active_honchkrow and discard_supporters > 0 and hand_supporters < 3 else 300
                elif identifier == HONCHKROW:
                    score += 5_400 if energy_murkrow_needs_honchkrow else 900
                elif identifier == FACTORY:
                    if supporter_played and FACTORY not in stadium_ids:
                        score += 36_000
                    elif ARIANA not in hand_ids and FACTORY not in stadium_ids:
                        score += 9_000
                elif identifier == ARIANA:
                    score += 10_000 if not proton_opening_allowed else 1_400
                    score += 1_400 if athena_draw_count >= 3 or hand_count <= 5 else 560
                    if energy_dig_needed:
                        score += _policy_rule_number("preferArianaEnergyDig", "pokegearArianaBonus", 5_200)
                elif identifier == PROTON:
                    if proton_opening_allowed:
                        score += 1_100
                    else:
                        score -= _policy_rule_number("preferProtonWhenSetupIncomplete", "settledSearchPenalty", 50_000)
                elif identifier == PETREL:
                    score += 800
                elif identifier == GIOVANNI:
                    score = giovanni_fuel_search_score()
                elif identifier == ARCHER:
                    score = _apollo_search_score(observation)
            elif effect == NIGHT_STRETCHER:
                score = _night_stretcher_target_score(observation, option_card)
            if score > 0 and (best is None or score > best[0] or (score == best[0] and option_index < best[1])):
                best = (score, option_index, option, identifier)

        if best is None:
            break

        _, option_index, option, identifier = best
        selected.append(option_index)
        if identifier is not None:
            counts[identifier] = counts.get(identifier, 0) + 1
        available = [(idx, opt) for idx, opt in available if idx != option_index]

    return selected


def _choose_optional_multi_select(observation, options, max_count):
    if max_count <= 0:
        return []
    taunt_lock = _choose_taunt_move_lock_option(observation, options, max_count)
    if taunt_lock is not None:
        return taunt_lock
    rocket_feather_costs = _choose_rocket_feather_costs(observation, options, max_count)
    if rocket_feather_costs is not None:
        return rocket_feather_costs
    return _choose_optional_cards(observation, options, max_count)


def _main_action_score(observation, option, turn_plan=None):
    current, your_index, player = _current_player(observation)
    if turn_plan is None:
        turn_plan = _build_donkrow_turn_plan(observation)
    option_type = _option_type(option)
    card = _card_from_option(observation, option)
    identifier = _card_id(card)
    target = _target_card_from_option(observation, option)
    target_id = _card_id(target)
    promotion_card = _promotion_card_from_option(observation, option)
    promotion_id = _card_id(promotion_card)
    field_ids = _field_card_ids(player)
    field_top_cards = _field_top_cards(player)
    field_top_ids = [_card_id(field_card) for field_card in field_top_cards]
    hand_ids = [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]
    deck_ids = [_card_id(card) for card in _iter_cards(_read(player, "deck", []))]
    active_ids = [_card_id(card) for card in _iter_cards(_read(player, "active", []))]
    active_card = _top_card(_read(player, "active", []))
    hand_count = _hand_count(player)
    hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    hand_energy = _count_cards(player, ("hand",), lambda card_id: card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY))
    deck_energy = _count_cards(player, ("deck",), lambda card_id: card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY))
    discard_supporters = _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    deck_count = _deck_count(player)
    opponent = _player_state(current, 1 - your_index)
    self_prizes = _prize_cards_remaining(player)
    opponent_prizes = _prize_cards_remaining(opponent)
    murkrow_lines = field_ids.count(MURKROW) + field_ids.count(HONCHKROW)
    porygon_lines = field_ids.count(PORYGON) + field_ids.count(PORYGON2)
    bench_count = _zone_count(player, "bench")
    attack_id = _attack_id(option)
    has_rocket_feather_action = _has_main_attack_option(observation, ROCKET_FEATHER_ATTACK)
    has_switch_option = _has_switch_option(observation)
    has_any_attack_option = _has_any_main_attack_option(observation)
    active_honchkrow = any(card_id == HONCHKROW for card_id in active_ids)
    active_porygon2 = any(card_id == PORYGON2 for card_id in active_ids)
    active_has_ignition_energy = IGNITION_ENERGY in _attached_energy_card_ids(active_card)
    active_attack_ready = (
        (active_honchkrow and has_rocket_feather_action and hand_supporters > 0) or
        (active_porygon2 and has_any_attack_option)
    )
    field_has_ready_attacker = any(card_id in (HONCHKROW, PORYGON2) for card_id in field_ids)
    field_honchkrow_needs_energy = any(
        _card_id(card) == HONCHKROW and _attached_energy_cards(card) <= 0
        for card in _field_cards(player)
    )
    murkrow_can_become_honchkrow = HONCHKROW in hand_ids or HONCHKROW in deck_ids
    ready_bench_attackers = _ready_bench_attacker_ids(player)
    all_rocket_field = _all_field_pokemon_are_rocket(player)
    athena_draw_count = _ariana_draw_count_for_player(player)
    supporter_played = _supporter_played_this_turn(current, player)
    stadium_played = _stadium_played_this_turn(current, player)
    stadium_ids = [_card_id(card) for card in _iter_cards(_read(current, "stadium", []))]
    target_energy_units = _attached_energy_units(target)
    target_energy_cards = _attached_energy_cards(target)
    target_energy_ids = _attached_energy_card_ids(target)
    target_has_rocket_energy = TEAM_ROCKET_ENERGY in target_energy_ids
    target_is_active = _area_code(_read(option, "inPlayArea"), -1) == 4
    target_is_bench = _area_code(_read(option, "inPlayArea"), -1) == 5
    rocket_feather_fuel_need = active_honchkrow and (hand_supporters < 3 or discard_supporters > 0)
    setup_incomplete = bench_count < 2 or murkrow_lines < 2 or porygon_lines < 1
    proton_opening_allowed = _proton_opening_allowed(current, player, setup_incomplete)
    turn_one_proton_priority = _turn_one_proton_priority(current, player)
    early_turn = _safe_int(_read(current, "turn"), 0) <= _policy_rule_number("preferProtonWhenSetupIncomplete", "earlyTurnMax", 4)
    opponent_active_hp = _remaining_hp(_opponent_active_card(observation))
    ariana_compression_available = _has_ariana_compression_option(observation)
    energy_murkrow_needs_honchkrow = _energy_prepared_murkrow_needs_honchkrow(player, hand_ids, deck_ids)
    energy_dig_needed = _needs_ariana_energy_dig(player)
    needs_honchkrow_from_pad, needs_porygon2_from_pad, active_pad_evolution = _poke_pad_evolution_attack_need(player, hand_ids)
    porygon_development_allowed = _porygon_development_allowed(player)
    petrel_attack_bridge = energy_murkrow_needs_honchkrow or needs_honchkrow_from_pad or needs_porygon2_from_pad or active_pad_evolution
    urgent_attack_window = opponent_prizes <= 2 or self_prizes <= 2 or deck_count <= 8
    ariana_desired = (
        ARIANA in hand_ids
        and not supporter_played
        and athena_draw_count > 0
        and (
            energy_dig_needed
            or hand_count <= 5
            or athena_draw_count >= 3
            or (has_rocket_feather_action and hand_supporters < 3)
        )
    )
    ariana_compression_needed = (
        ariana_desired
        and _policy_rule_enabled("compressBeforeAriana")
        and hand_count >= _policy_rule_number("compressBeforeAriana", "compressionHandMin", 6)
        and ariana_compression_available
    )
    factory_pending_after_supporter = supporter_played and not stadium_played and _factory_option_available(observation)

    if option_type in (14, "end"):
        return -20_000

    seed_guard_play_score = _seed_out_basic_play_score(observation, option)
    seed_guard_has_basic = _has_basic_play_option(observation)
    thin_board_guard = _needs_thin_board_basic_guard(player)
    basic_continuity_option = _has_basic_continuity_option(observation)

    if promotion_id is not None and _policy_rule_enabled("preferHonchkrowPromotion"):
        chain_promotion_score = _honchkrow_chain_promotion_score(
            player,
            promotion_id,
            promotion_card,
            hand_ids,
            deck_ids,
            hand_supporters,
        )
        if chain_promotion_score is not None:
            return chain_promotion_score
        if promotion_id == HONCHKROW:
            return _policy_rule_number("preferHonchkrowPromotion", "honchkrowScore", 26_000)
        if promotion_id == PORYGON2:
            return _policy_rule_number("preferHonchkrowPromotion", "porygon2Score", 22_500)
        if promotion_id == MURKROW and field_has_ready_attacker:
            return _policy_rule_number("preferHonchkrowPromotion", "murkrowWithAttackerPenalty", -18_000)
        if promotion_id == PORYGON and field_has_ready_attacker:
            return _policy_rule_number("preferHonchkrowPromotion", "porygonWithAttackerPenalty", -12_000)
        return _policy_rule_number("preferHonchkrowPromotion", "fallbackScore", 1_200)

    if option_type in (13, "attack"):
        if seed_guard_has_basic and _needs_seed_out_bench_guard(player):
            return -75_000
        if thin_board_guard and basic_continuity_option and not (turn_plan is not None and turn_plan.game_end):
            if attack_id in MURKROW_ATTACKS:
                return -82_000
            if attack_id == ROCKET_FEATHER_ATTACK and opponent_active_hp > 0:
                damage_per_supporter = _rocket_feather_damage_per_supporter(_opponent_active_card(observation))
                required_for_ko = max(1, (opponent_active_hp + damage_per_supporter - 1) // damage_per_supporter)
                if hand_supporters < required_for_ko and not (active_has_ignition_energy and hand_supporters > 0):
                    return -54_000
        if attack_id == ROCKET_FEATHER_ATTACK:
            planned_score = (
                _turn_plan_attack_score(turn_plan)
                if turn_plan is not None and turn_plan.attacker_index == 0 and turn_plan.attack_id == ROCKET_FEATHER_ATTACK
                else 0
            )
            if hand_supporters <= 0:
                return _policy_rule_number("avoidNonKoRocketFeatherIntoHealing", "zeroFuelScore", -22_000)
            damage_per_supporter = _rocket_feather_damage_per_supporter(_opponent_active_card(observation))
            required_for_ko = max(1, (opponent_active_hp + damage_per_supporter - 1) // damage_per_supporter) if opponent_active_hp > 0 else 1
            can_ko = opponent_active_hp > 0 and hand_supporters >= required_for_ko
            if can_ko:
                score = (
                    _policy_rule_number("avoidNonKoRocketFeatherIntoHealing", "koBaseScore", 16_500) +
                    min(required_for_ko, 4) *
                    _policy_rule_number("avoidNonKoRocketFeatherIntoHealing", "koPerRequiredSupporterScore", 700)
                )
                if damage_per_supporter > 60:
                    score += 3_200
                if deck_count <= 5:
                    score += _policy_rule_number("avoidNonKoRocketFeatherIntoHealing", "lowDeckKoBonus", 2_400)
                return max(score, planned_score)
            missing_for_ko = max(0, required_for_ko - hand_supporters)
            score = (
                _policy_rule_number("avoidNonKoRocketFeatherIntoHealing", "baseNonKoScore", 900) +
                min(hand_supporters, 3) *
                _policy_rule_number("avoidNonKoRocketFeatherIntoHealing", "perSupporterNonKoScore", 400) -
                missing_for_ko *
                _policy_rule_number("avoidNonKoRocketFeatherIntoHealing", "missingKoSupporterPenalty", 1_800)
            )
            if hand_supporters <= 1:
                score -= _policy_rule_number("avoidNonKoRocketFeatherIntoHealing", "oneSupporterPenalty", 4_800)
            if opponent_active_hp >= 120:
                score -= _policy_rule_number("avoidNonKoRocketFeatherIntoHealing", "hp120Penalty", 2_800)
            if opponent_active_hp >= 180:
                score -= _policy_rule_number("avoidNonKoRocketFeatherIntoHealing", "hp180Penalty", 2_000)
            if deck_count <= _policy_rule_number("avoidNonKoRocketFeatherIntoHealing", "lowDeckThreshold", 10):
                score -= _policy_rule_number("avoidNonKoRocketFeatherIntoHealing", "lowDeckPenalty", 5_600)
            if setup_incomplete or has_switch_option:
                score -= _policy_rule_number("avoidNonKoRocketFeatherIntoHealing", "setupOrSwitchPenalty", 1_600)
            if planned_score and turn_plan is not None and (turn_plan.can_ko or turn_plan.game_end):
                return max(score, planned_score)
            if active_has_ignition_energy and hand_supporters > 0 and not factory_pending_after_supporter:
                score = max(score, 1_800 + min(hand_supporters, 3) * 900)
            if hand_supporters >= 2 and not factory_pending_after_supporter:
                score = max(score, 2_400 + min(hand_supporters, 3) * 500)
            if urgent_attack_window and not factory_pending_after_supporter:
                attack_floor = 5_200 + min(hand_supporters, 3) * 900
                if opponent_prizes <= 1:
                    attack_floor += 3_600
                if self_prizes <= 2:
                    attack_floor += 2_400
                if opponent_active_hp <= damage_per_supporter * max(1, hand_supporters):
                    attack_floor += 1_600
                score = max(score, attack_floor)
            return score
        if attack_id in PORYGON2_R_COMMAND_ATTACKS:
            planned_score = (
                _turn_plan_attack_score(turn_plan)
                if turn_plan is not None and turn_plan.attacker_index == 0 and turn_plan.attack_id in PORYGON2_R_COMMAND_ATTACKS
                else 0
            )
            damage = discard_supporters * _policy_rule_number("porygon2RCommandFallback", "damagePerTrashRocketSupporter", 20)
            if damage <= 0:
                return _policy_rule_number("porygon2RCommandFallback", "zeroDamageScore", -12_000)
            porygon2_bridge_window = (
                _safe_int(_read(current, "turn"), 0) >= 6
                or _prize_cards_remaining(player) <= 4
                or deck_count <= 20
            )
            porygon2_bridge_bonus = 0
            if porygon2_bridge_window and damage >= 80:
                porygon2_bridge_bonus += 4_800
                if damage >= 100:
                    porygon2_bridge_bonus += 3_400
                if damage >= 120:
                    porygon2_bridge_bonus += 2_800
                if not has_rocket_feather_action:
                    porygon2_bridge_bonus += 4_200
                elif hand_supporters <= 1:
                    porygon2_bridge_bonus += 1_800
            if opponent_active_hp > 0 and damage >= opponent_active_hp:
                score = (
                    _policy_rule_number("porygon2RCommandFallback", "koScore", 14_800) +
                    min(discard_supporters, 6) *
                    _policy_rule_number("porygon2RCommandFallback", "koPerSupporterScore", 250)
                )
                score += porygon2_bridge_bonus
                return max(score, planned_score)
            score = (
                _policy_rule_number("porygon2RCommandFallback", "nonKoBaseScore", 6_200) +
                min(discard_supporters, 6) *
                _policy_rule_number("porygon2RCommandFallback", "nonKoPerSupporterScore", 520)
            )
            score += min(damage, 180) * 18
            score += porygon2_bridge_bonus
            if deck_count <= _policy_rule_number("porygon2RCommandFallback", "lowDeckThreshold", 10):
                score -= _policy_rule_number("porygon2RCommandFallback", "lowDeckPenalty", 2_800)
            if urgent_attack_window and damage >= 40:
                attack_floor = 5_400 + min(discard_supporters, 6) * 550
                if opponent_prizes <= 1:
                    attack_floor += 3_200
                if self_prizes <= 2:
                    attack_floor += 2_000
                score = max(score, attack_floor)
            return max(score, planned_score)
        if attack_id == TAUNT_ATTACK:
            threat = _opponent_active_attack_threat(observation)
            if threat <= 0:
                return _policy_rule_number("auxiliaryTauntPlan", "notReadyPenalty", -12_000)
            score = (
                _policy_rule_number("auxiliaryTauntPlan", "readyOpponentScore", 3_600) +
                threat * _policy_rule_number("auxiliaryTauntPlan", "damageBonus", 18)
            )
            if setup_incomplete or hand_energy <= 0:
                score -= _policy_rule_number("auxiliaryTauntPlan", "setupPenalty", 1_400)
            return score
        if attack_id in MURKROW_ATTACKS:
            if attack_id == MURKROW_TEMPT_ATTACK:
                if hand_supporters > 0:
                    return -220_000
                if any(card_id in (TEAM_ROCKET_TRANSCEIVER, POKEGEAR, POKE_PAD, ROTO_STICK, MIRACLE_HEADSET) for card_id in hand_ids):
                    return -72_000
                score = 3_800
                if setup_incomplete:
                    score += 900
                if energy_dig_needed:
                    score += 1_100
                if deck_count <= 4:
                    score -= 1_400
                return score
            return -120_000
        return 2_200

    if option_type in (9, "evolve"):
        if seed_guard_has_basic and _needs_seed_out_bench_guard(player):
            return -42_000
        if ariana_compression_needed and _is_ariana_compression_option(observation, option):
            return 28_000
        if identifier == HONCHKROW:
            score = 10_500
            if target_id == MURKROW:
                score = (
                    _policy_rule_number("attackContinuity", "activeHonchkrowEvolutionScore", 42_000)
                    if target_is_active
                    else _policy_rule_number("attackContinuity", "benchHonchkrowEvolutionScore", 28_000)
                )
                if target_energy_cards > 0:
                    score += 9_000
                elif hand_energy > 0:
                    score += 6_500
                if hand_supporters > 0:
                    score += 4_200
                if target_is_active and (target_energy_cards > 0 or hand_energy > 0) and hand_supporters > 0:
                    score += 5_600
            return score
        if identifier == PORYGON2:
            if not porygon_development_allowed:
                return 1_800
            if _honchkrow_chain_available(player, hand_ids, deck_ids) and not _only_porygon_board(player):
                return 2_400
            if _only_porygon_board(player):
                return 14_800 if IGNITION_ENERGY in [_card_id(card) for card in _iter_cards(_read(player, "hand", []))] else 11_400
            if target_id == PORYGON and target_is_active and IGNITION_ENERGY in hand_ids and not _honchkrow_chain_available(player, hand_ids, deck_ids):
                return 24_000
            return 11_200 if IGNITION_ENERGY in [_card_id(card) for card in _iter_cards(_read(player, "hand", []))] else 8_600
        return 4_000

    if option_type in (8, "attach"):
        if (
            turn_plan is not None
            and not turn_plan.seed_guard_blocked
            and turn_plan.needs_energy
            and _option_targets_plan_attacker(target, turn_plan)
        ):
            plan_attach_score = _policy_rule_number("donkrowTurnPlan", "energyAttachScore", 26_000) + max(0, turn_plan.score // 20)
            if identifier == TEAM_ROCKET_ENERGY and turn_plan.attacker_id in (HONCHKROW, MURKROW):
                if turn_plan.attacker_id == HONCHKROW:
                    return plan_attach_score + 2_400
                return plan_attach_score
            if identifier == IGNITION_ENERGY and turn_plan.attacker_id in (HONCHKROW, PORYGON2):
                if target_is_active or turn_plan.needs_switch or has_switch_option:
                    if turn_plan.attacker_id == HONCHKROW and TEAM_ROCKET_ENERGY in hand_ids:
                        return 600
                    return plan_attach_score
        if identifier == TEAM_ROCKET_ENERGY:
            if target_has_rocket_energy:
                return _policy_rule_number("avoidMurkrowEnergyWhenAttackerReady", "duplicateRocketEnergyPenalty", -20_000)
            if target_id == HONCHKROW:
                if target_energy_units >= 2 or target_energy_cards >= 1:
                    return -16_000
                score = 10_800
                if target_is_active:
                    score += 1_200
                if active_attack_ready and target_is_bench:
                    score += _policy_rule_number("attackContinuity", "nextAttackerRocketEnergyBonus", 21_000)
                    if hand_supporters > 0:
                        score += 4_400
                elif active_honchkrow and target_id == HONCHKROW and not target_is_active:
                    score += 8_800
                return score
            if target_id == MURKROW:
                if target_energy_cards >= 1:
                    return _policy_rule_number("avoidMurkrowEnergyWhenAttackerReady", "duplicateRocketEnergyPenalty", -20_000)
                if _policy_rule_enabled("avoidMurkrowEnergyWhenAttackerReady") and field_honchkrow_needs_energy:
                    return _policy_rule_number("avoidMurkrowEnergyWhenAttackerReady", "existingHonchkrowNeedsEnergyPenalty", -15_000)
                if murkrow_can_become_honchkrow:
                    score = _policy_rule_number("avoidMurkrowEnergyWhenAttackerReady", "evolveableMurkrowScore", 7_600)
                    if field_has_ready_attacker:
                        score += _policy_rule_number("avoidMurkrowEnergyWhenAttackerReady", "nextAttackerPreparationBonus", 2_400)
                    if active_attack_ready and target_is_bench:
                        score += _policy_rule_number("attackContinuity", "nextAttackerRocketEnergyBonus", 21_000)
                        if HONCHKROW in hand_ids:
                            score += 5_200
                        if hand_supporters > 0:
                            score += 2_800
                    elif active_honchkrow and has_rocket_feather_action:
                        score += 5_200
                    return score
                if _policy_rule_enabled("avoidMurkrowEnergyWhenAttackerReady") and field_has_ready_attacker:
                    return _policy_rule_number("avoidMurkrowEnergyWhenAttackerReady", "betterAttackerPenalty", -17_000)
                if active_honchkrow and has_rocket_feather_action:
                    return _policy_rule_number("avoidMurkrowEnergyWhenAttackerReady", "activeAttackReadyMurkrowScore", 1_800)
                return _policy_rule_number("avoidMurkrowEnergyWhenAttackerReady", "earlyMurkrowScore", 5_200) if not active_honchkrow else 4_000
            if target_id in (PORYGON, PORYGON2):
                return -16_000
            return 1_000
        if identifier == IGNITION_ENERGY:
            planned_ignition_attack = (
                turn_plan is not None
                and turn_plan.needs_energy
                and turn_plan.attacker_id in (HONCHKROW, PORYGON2)
                and _option_targets_plan_attacker(target, turn_plan)
            )
            if not planned_ignition_attack and (not target_is_active or not has_any_attack_option):
                return _policy_rule_number("avoidIgnitionWaste", "forbiddenTargetScore", -16_000)
            if target_id == HONCHKROW:
                if not _policy_rule_bool("avoidIgnitionWaste", "allowImmediateIgnitionOnHonchkrow", True):
                    return _policy_rule_number("avoidIgnitionWaste", "forbiddenTargetScore", -16_000)
                if target_energy_cards >= 1 or has_rocket_feather_action:
                    return _policy_rule_number("avoidIgnitionWaste", "forbiddenTargetScore", -16_000)
                if TEAM_ROCKET_ENERGY in hand_ids:
                    return 600
                if hand_supporters <= 0:
                    return 700
                if not target_is_active and not has_switch_option:
                    return _policy_rule_number("avoidIgnitionWaste", "benchWithoutSwitchPenalty", -15_000)
                score = _policy_rule_number("avoidIgnitionWaste", "honchkrowActiveAttackScore", 15_800) if target_is_active else _policy_rule_number("avoidIgnitionWaste", "honchkrowSwitchableAttackScore", 10_200)
                required_for_ko = max(1, (opponent_active_hp + 59) // 60) if opponent_active_hp > 0 else 1
                if hand_supporters >= required_for_ko:
                    score += 9_500
                score += min(hand_supporters, 4) * 850
                return score
            if target_id == MURKROW and _policy_rule_bool("avoidIgnitionWaste", "forbidIgnitionOnBasicMurkrow", True):
                return _policy_rule_number("avoidIgnitionWaste", "forbiddenTargetScore", -16_000)
            if target_id == PORYGON2:
                if IGNITION_ENERGY in target_energy_ids:
                    return _policy_rule_number("avoidIgnitionWaste", "duplicateIgnitionPenalty", -11_000)
                if not target_is_active and not has_switch_option:
                    return _policy_rule_number("avoidIgnitionWaste", "benchWithoutSwitchPenalty", -15_000)
                score = _policy_rule_number("avoidIgnitionWaste", "porygon2ActiveScore", 11_800) if target_is_active else _policy_rule_number("avoidIgnitionWaste", "porygon2SwitchableScore", 8_600)
                if active_honchkrow and has_rocket_feather_action and hand_supporters > 0:
                    score -= 3_800
                return score
            if target_id == PORYGON:
                return _policy_rule_number("avoidIgnitionWaste", "forbiddenTargetScore", -16_000)
            if target_id in (HONCHKROW, MURKROW) and _policy_rule_bool("avoidIgnitionWaste", "forbidIgnitionOnMurkrowOrHonchkrow", False):
                return _policy_rule_number("avoidIgnitionWaste", "forbiddenTargetScore", -16_000)
        return 2_500

    if option_type in (7, "play"):
        if seed_guard_play_score is not None:
            return seed_guard_play_score
        if ariana_compression_needed and _is_ariana_compression_option(observation, option):
            return 26_000
        if identifier == MURKROW:
            if active_attack_ready and bench_count < _bench_limit(player):
                if field_top_ids.count(MURKROW) + field_top_ids.count(HONCHKROW) < 3:
                    return _policy_rule_number("attackContinuity", "activeReadyBasicBenchScore", 32_000)
                return 12_500
            if thin_board_guard:
                return 118_000 if len(field_ids) <= 1 else 84_000
            return 9_300 if murkrow_lines < 3 else 3_200
        if identifier == PORYGON:
            if active_attack_ready and bench_count < _bench_limit(player) and PORYGON not in field_top_ids and PORYGON2 not in field_top_ids:
                return 18_000
            if thin_board_guard:
                return 94_000 if len(field_ids) <= 1 else 64_000
            return 7_700 if porygon_lines < 1 else 2_700
        if identifier == HONCHKROW:
            return 7_200 if murkrow_lines > 0 else 900
        if identifier == PORYGON2:
            if not porygon_development_allowed:
                return 800
            if _honchkrow_chain_available(player, hand_ids, deck_ids) and not _only_porygon_board(player):
                return 600
            if _only_porygon_board(player):
                return 8_800 if porygon_lines > 0 else 1_200
            return 5_800 if porygon_lines > 0 else 800
        if identifier == PROTON:
            if supporter_played:
                return -7_500
            if turn_one_proton_priority:
                return 300_000
            if proton_opening_allowed:
                return _policy_rule_number("preferProtonWhenSetupIncomplete", "mainEarlyScore", 14_400) if early_turn else _policy_rule_number("preferProtonWhenSetupIncomplete", "mainSetupScore", 11_600)
            return _policy_rule_number("preferProtonWhenSetupIncomplete", "settledMainScore", -50_000)
        if identifier == PETREL:
            if supporter_played:
                return -7_500
            petrel_factory_bridge = ARIANA not in hand_ids and FACTORY not in stadium_ids
            score = 1_400
            if petrel_attack_bridge:
                score += 8_600
            if petrel_factory_bridge:
                score += 6_200
            if rocket_feather_fuel_need and active_honchkrow:
                score += 1_700
            if energy_murkrow_needs_honchkrow:
                score += 2_900
            if not petrel_attack_bridge and not petrel_factory_bridge:
                score -= 6_800
            if energy_dig_needed:
                score -= 7_800 if ARIANA in hand_ids else 2_800
            if setup_incomplete and (PROTON in hand_ids or TEAM_ROCKET_TRANSCEIVER in hand_ids or POKE_PAD in hand_ids):
                score -= 3_200
            if hand_count <= 5 and hand_energy <= 0:
                score -= 2_400
            return score
        if identifier == TEAM_ROCKET_TRANSCEIVER:
            if turn_one_proton_priority and PROTON not in hand_ids:
                return 240_000
            if PROTON in hand_ids and not supporter_played and proton_opening_allowed:
                return 3_200
            score = 9_900 if setup_incomplete else 5_000
            if energy_murkrow_needs_honchkrow:
                score += 4_900
            if energy_dig_needed:
                score += _policy_rule_number("preferArianaEnergyDig", "transceiverArianaBonus", 8_800)
            return score + (4_300 if rocket_feather_fuel_need else 0)
        if identifier == POKE_PAD:
            pad_score = (13_800 if energy_murkrow_needs_honchkrow else 9_200) if setup_incomplete else (9_800 if energy_murkrow_needs_honchkrow else 4_600)
            pad_score = max(pad_score, _recovery_tool_use_score(_best_poke_pad_target_score(observation)))
            if active_attack_ready and (field_top_ids.count(MURKROW) + field_top_ids.count(HONCHKROW) < 3 or _needs_thin_board_basic_guard(player)):
                pad_score = max(pad_score, _policy_rule_number("attackContinuity", "activeReadyPokePadScore", 30_000))
            if needs_honchkrow_from_pad:
                pad_score += _policy_rule_number("pokePadEvolutionAttack", "evolutionBackupScore", 5_600)
            if needs_porygon2_from_pad:
                pad_score += _policy_rule_number("pokePadEvolutionAttack", "evolutionBackupScore", 5_600)
            if active_pad_evolution:
                pad_score += _policy_rule_number("pokePadEvolutionAttack", "activeEvolutionBonus", 2_800)
            return pad_score
        if identifier == POKEGEAR:
            if turn_one_proton_priority and PROTON not in hand_ids:
                return 235_000
            return 8_800 + (3_900 if rocket_feather_fuel_need else 0) + (2_200 if energy_murkrow_needs_honchkrow else 0) + (
                _policy_rule_number("preferArianaEnergyDig", "pokegearArianaBonus", 5_200) if energy_dig_needed else 0
            )
        if identifier == FACTORY:
            if stadium_played:
                return -6_500
            if supporter_played:
                return 72_000
            if ariana_desired or (not supporter_played and energy_dig_needed and ARIANA in hand_ids):
                return -_policy_rule_number("preferArianaEnergyDig", "factoryBeforeArianaPenalty", 6_000)
            return 10_400 if supporter_played else 3_100
        if identifier == ROTO_STICK:
            if turn_one_proton_priority and PROTON not in hand_ids:
                return 230_000
            return 7_900 + (4_200 if rocket_feather_fuel_need else 0) + (3_200 if energy_murkrow_needs_honchkrow else 0) + (
                _policy_rule_number("preferArianaEnergyDig", "pokegearArianaBonus", 5_200) if energy_dig_needed else 0
            )
        if identifier == NIGHT_STRETCHER:
            return _recovery_tool_use_score(_best_night_stretcher_target_score(observation))
        if identifier == MIRACLE_HEADSET:
            if ariana_desired and not supporter_played:
                return -9_000
            if discard_supporters >= 2 or (discard_supporters > 0 and active_honchkrow and hand_supporters <= 1):
                score = 6_800 + min(2, discard_supporters) * 2_200
                if active_honchkrow:
                    score += 5_700
                if has_rocket_feather_action:
                    score += 4_800
                if hand_supporters < 3:
                    score += (3 - hand_supporters) * 2_600
                if discard_supporters >= 2:
                    score += 1_800
                return score
            return -9_000
        if identifier == ARIANA:
            if supporter_played or athena_draw_count <= 0:
                return -7_500
            if ariana_compression_needed:
                return -18_000
            score = 2_600 + athena_draw_count * 650
            if all_rocket_field:
                score += 600
            if field_ids:
                score += 8_200
            if _safe_int(_read(current, "turn"), 0) >= 2:
                score += 7_500
            if hand_count <= 4 or athena_draw_count >= 4:
                score += 5_400
            if hand_energy <= 0 and deck_energy > 0:
                score += _policy_rule_number("preferArianaEnergyDig", "noHandEnergyBonus", 9_200)
            if active_honchkrow:
                score += 1_400
            if active_honchkrow and hand_supporters < 2:
                score += 7_500
            if has_rocket_feather_action and hand_supporters < 3:
                score += 3_500
            if FACTORY in [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]:
                score += 1_300
            score += max(0, 3 - hand_supporters) * 1_350
            score += min(discard_supporters, 2) * 500
            if energy_dig_needed:
                score += _policy_rule_number("preferArianaEnergyDig", "attackStarvedBonus", 10_500)
            if not proton_opening_allowed:
                score += 10_000
            if _policy_rule_enabled("compressBeforeAriana"):
                if athena_draw_count <= _policy_rule_number("compressBeforeAriana", "lowDrawMax", 2) and hand_count >= _policy_rule_number("compressBeforeAriana", "highHandMin", 7):
                    score -= _policy_rule_number("compressBeforeAriana", "lowDrawPenalty", 12_000)
                if ariana_compression_available and hand_count >= _policy_rule_number("compressBeforeAriana", "compressionHandMin", 6):
                    score -= _policy_rule_number("compressBeforeAriana", "compressionPenalty", 8_500)
            if proton_opening_allowed and early_turn and (PROTON in hand_ids or TEAM_ROCKET_TRANSCEIVER in hand_ids or POKE_PAD in hand_ids):
                score -= 5_400
            return score
        if identifier == GIOVANNI:
            if supporter_played:
                return -7_500
            sakaki_ko_score = _sakaki_two_prize_ko_score(observation, giovanni_from_hand=True)
            if sakaki_ko_score is not None:
                return sakaki_ko_score
            return _policy_rule_number("sakakiRequiresKo", "nonKoScore", -9_000)
        if identifier == ARCHER:
            if supporter_played:
                return -7_500
            apollo_score = _apollo_reset_score(observation, assume_legal=True, apollo_from_hand=True)
            if apollo_score > 0:
                if energy_dig_needed and ARIANA in hand_ids:
                    apollo_score -= 6_000
                return apollo_score
            return apollo_score
        return 500

    if option_type in (10, "ability"):
        if identifier == FACTORY:
            if stadium_played:
                return -6_500
            if supporter_played:
                return 70_000
            if ariana_desired:
                return -6_000
            return 1_000
        return 5_700

    if option_type in (12, "retreat", "switch", "promote"):
        if not _policy_rule_enabled("avoidLowValueSwitching"):
            return 1_400
        if (
            turn_plan is not None
            and not turn_plan.seed_guard_blocked
            and turn_plan.needs_switch
            and _switch_option_targets_plan_attacker(option, target, promotion_card, turn_plan)
        ):
            return _policy_rule_number("donkrowTurnPlan", "switchScore", 30_000) + max(0, turn_plan.score // 30)
        if active_attack_ready:
            return _policy_rule_number("avoidLowValueSwitching", "activeAttackerPenalty", -22_000)
        if ready_bench_attackers:
            score = _policy_rule_number("avoidLowValueSwitching", "readyBenchAttackerScore", 7_200)
            if PORYGON2 in ready_bench_attackers:
                score += 2_400
            if HONCHKROW in ready_bench_attackers:
                score += 1_800
            return score
        return _policy_rule_number("avoidLowValueSwitching", "noReadyBenchAttackerPenalty", -18_000)

    return 0


def _choose_donkrow_main_action(observation, options):
    select = _read(observation, "select", {})
    if _select_context(select) not in (0, "main"):
        return None

    current, _, player = _current_player(observation)
    supporter_played = _supporter_played_this_turn(current, player)

    if supporter_played:
        for option_index, option in enumerate(options):
            if _card_id(_card_from_option(observation, option)) == FACTORY and _option_type(option) in (10, "ability"):
                return option_index
            
    has_athena_option = any(
        _card_id(_card_from_option(observation, option)) == ARIANA
        for option in options
    )

    if has_athena_option:
        for option_index, option in enumerate(options):
            identifier = _card_id(_card_from_option(observation, option))
            option_type = _option_type(option)
            if identifier == FACTORY and option_type in (7, 10, "play", "ability"):
                return option_index

        for option_index, option in enumerate(options):
            identifier = _card_id(_card_from_option(observation, option))
            option_type = _option_type(option)
            if option_type in (9, "evolve") and identifier in (HONCHKROW, PORYGON2):
                return option_index

        for option_index, option in enumerate(options):
            if _option_type(option) in (8, "attach"):
                target = _target_card_from_option(observation, option)
                if _card_id(target) in (HONCHKROW, PORYGON2):
                    return option_index

    turn_plan = _build_donkrow_turn_plan(observation)
    scored = []
    for option_index, option in enumerate(options):
        scored.append((_main_action_score(observation, option, turn_plan), option_index))
    if not scored:
        return None

    scored.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_index = scored[0]
    if best_score > 0:
        return best_index
    if best_score == 0:
        return None
    for score, option_index in scored:
        if _option_type(options[option_index]) in (14, "end"):
            return option_index
    return None


def _sanitize_action(indices, min_count, max_count, option_count):
    result = []
    for index in indices:
        if isinstance(index, bool):
            continue
        if isinstance(index, int) and 0 <= index < option_count and index not in result:
            result.append(index)
        if len(result) >= max_count:
            break

    if len(result) < min_count:
        for index in range(option_count):
            if index not in result:
                result.append(index)
            if len(result) >= min_count:
                break

    return result[:max_count]


def agent(obs_dict):
    select = _select_payload(obs_dict)
    if _is_initial_deck_request(obs_dict, select):
        return MY_DECK

    options = _select_options(select)
    min_count, max_count = _selection_bounds(select)
    if max_count <= 0 or not options:
        return []

    context = _select_context(select)
    if context not in (0, "main"):
        selected_cards = _choose_optional_multi_select(obs_dict, options, max_count)
        if selected_cards or min_count == 0:
            return _sanitize_action(selected_cards, min_count, max_count, len(options))

    special = _choose_donkrow_main_action(obs_dict, options)
    if isinstance(special, int):
        return _sanitize_action([special], min_count, max_count, len(options))

    selected = choose_action(obs_dict, "public", "donkrow")
    if not isinstance(selected, int) or selected < 0 or selected >= len(options):
        return [0] if min_count > 0 else []

    if min_count == 1:
        return _sanitize_action([selected], min_count, max_count, len(options))

    ranked = rank_actions(obs_dict, "public", "donkrow")
    ordered = [item["index"] for item in ranked if isinstance(item.get("index"), int)]
    ordered = [index for index in ordered if 0 <= index < len(options)]
    target_count = min(max(min_count, 1), max_count, len(options))
    result = []
    for index in ordered:
        if index not in result:
            result.append(index)
        if len(result) >= target_count:
            break
    while len(result) < target_count:
        fallback = len(result)
        if fallback not in result and fallback < len(options):
            result.append(fallback)
        else:
            break
    return _sanitize_action(result, min_count, max_count, len(options))


class Agent:
    """Small adapter around the BOSS-derived policy."""

    def __init__(self, deck_profile="donkrow"):
        self.deck_profile = deck_profile

    def __call__(self, observation, *args, **kwargs):
        return agent(observation)

    def act(self, observation, *args, **kwargs):
        return agent(observation)

    def get_action(self, observation, *args, **kwargs):
        return agent(observation)


agent_instance = Agent()


def select_action(observation, *args, **kwargs):
    return agent(observation)


def get_action(observation, *args, **kwargs):
    return agent(observation)


def act(observation, *args, **kwargs):
    return agent(observation)
