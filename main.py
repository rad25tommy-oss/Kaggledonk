"""PTCG ABC submission entrypoint.

The simulator calls ``agent(obs_dict)``. On the very first call ``obs_dict`` has
``select == None`` and the agent must return the 60-card deck list. Later calls
must return a list of selected option indices.

This submission is intentionally self-contained: the BOSS-derived policy core
and the Donkrow policy overlay are embedded in this file so Kaggle/CABT output
only needs ``main.py`` plus ``deck.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import re
import sys
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
      "koBaseScore": 15047,
      "koPerRequiredSupporterScore": 700,
      "lowDeckKoBonus": 2400,
      "baseNonKoScore": 6200,
      "perSupporterNonKoScore": 1450,
      "missingKoSupporterPenalty": 900,
      "oneSupporterPenalty": 1200,
      "hp120Penalty": 900,
      "hp180Penalty": 600,
      "lowDeckThreshold": 10,
      "lowDeckPenalty": 3000,
      "setupOrSwitchPenalty": 600,
      "twoTurnKoProgressBonus": 11800,
      "ignitionTempoFloor": 52000,
      "safeNonKoAttackFloor": 18000,
      "bossReadyAttackPenalty": 70000,
      "bossRankPenalty": 90000
    },
    "compressBeforeAriana": {
      "enabled": true,
      "highHandMin": 7,
      "lowDrawMax": 2,
      "lowDrawPenalty": 11310,
      "compressionHandMin": 6,
      "compressionPenalty": 8453,
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
      "porygon2NonKoPenalty": 8000,
      "evolutionBridgeScore": 18000,
      "evolutionBridgeKoBonus": 22000,
      "evolutionBridgeTwoTurnBonus": 8000,
      "evolutionBridgeActionScore": 160000
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
      "lowDeckPenalty": 2800,
      "lateTrashThreshold": 15,
      "lateTrashPlanBonus": 52000,
      "lateTrashAttackBonus": 36000,
      "lateTrashKoBonus": 18000,
      "lateTrashPerExtraSupporterScore": 1400,
      "lateTrashHighDamageScore": 24,
      "endgamePrizeThreshold": 3,
      "endgameKoBonus": 46000,
      "endgameKoOverRocketFeatherBonus": 82000,
      "endgameLowDeckKoBonus": 24000
    },
    "alakazamLockPlan": {
      "deckPreserveThreshold": 20,
      "lowDeckReleaseThreshold": 8,
      "endgameReleasePrizeThreshold": 3
    },
    "loGuard": {
      "criticalDeckThreshold": 4,
      "nearDeckThreshold": 6,
      "criticalDrawPenalty": 220000,
      "nearDrawPenalty": 70000,
      "criticalSearchPenalty": 120000,
      "nearSearchPenalty": 32000,
      "criticalFactoryPenalty": 260000,
      "nearFactoryPenalty": 90000,
      "criticalSupportPenalty": 130000,
      "nearSupportPenalty": 36000
    },
    "preferArianaEnergyDig": {
      "enabled": true,
      "noHandEnergyBonus": 10925,
      "attackStarvedBonus": 10500,
      "transceiverArianaBonus": 5848,
      "pokegearArianaBonus": 5200,
      "petrelArianaBonus": 5815,
      "factoryBeforeArianaBonus": 34000,
      "factoryBeforeArianaDrawGainBonus": 8000,
      "factoryBeforeArianaPenalty": 6000
    },
    "pokePadEvolutionAttack": {
      "enabled": true,
      "honchkrowWithEnergyInHandScore": 12000,
      "porygon2WithIgnitionInHandScore": 11800,
      "evolutionBackupScore": 5961,
      "activeEvolutionBonus": 2800
    },
    "attackContinuity": {
      "enabled": true,
      "activeHonchkrowEvolutionScore": 42000,
      "benchHonchkrowEvolutionScore": 28000,
      "nextAttackerRocketEnergyBonus": 20316,
      "activeReadyBasicBenchScore": 32000,
      "activeReadyPokePadScore": 31329
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
      "transceiver": 248,
      "proton": -600,
      "petrel": 140,
      "ariana": 220,
      "archer": 30,
      "giovanni": 100,
      "factory": 148,
      "rotoStick": 70,
      "pokegear": 65,
      "pokePad": 90,
      "nightStretcher": 76,
      "miracleHeadset": 68
    },
    "board": {
      "benchMurkrow": 240,
      "benchMurkrowWhenNoAttack": 110,
      "benchPorygon": 130,
      "benchPorygonWhenNoAttack": 80,
      "benchRocketPokemon": 70,
      "evolveHonchkrow": 300,
      "evolveHonchkrowWithEnergy": 71,
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
      "openSupporterRight": 180,
      "usedSupporterPenalty": -280,
      "preserveAthenaPrizeRaceAhead": 380
    },
    "energy": {
      "openEnergyRight": 85,
      "usedEnergyPenalty": -450,
      "prepareWhenNoAttack": 331,
      "prepareBenchWhenNoAttack": 206,
      "immediateAttack": 323,
      "immediateKo": 275,
      "futurePlan": 160,
      "proactivePlan": 70,
      "honchkrowTarget": 240,
      "porygonAttackerTarget": 210,
      "murkrowTarget": 48,
      "murkrowNonImmediatePenalty": -142,
      "porygonBasicTarget": 55,
      "teamRocketEnergyOnPorygonPenalty": -260,
      "ignitionOnPorygonAttacker": 420,
      "porygonNonIgnitionPenalty": -720,
      "benchRocketEnergyWhileAttacking": 260,
      "honchkrowIgnitionRedundantPenalty": -900
    },
    "attack": {
      "rocketFeather": 420,
      "rocketFeatherKo": 269,
      "koOrPrize": 218,
      "tauntWithoutKoPenalty": -520,
      "nonKoBeforeSetupPenalty": -40,
      "rocketFeatherAthenaDiscardPrizeRacePenalty": -412,
      "rocketFeatherCostLance": 160,
      "rocketFeatherCostGiovanni": 151,
      "rocketFeatherCostArcher": 51,
      "rocketFeatherCostPetrel": 0,
      "rocketFeatherCostAthena": -80,
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
      "passWithAttack": -500
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
    4: "promote active pokemon",
    7: "to hand search",
    8: "rocket feather discard cost",
    38: "mulligan bonus draw",
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
ARTICUNO = 414
DREEPY = 119
DRAKLOAK = 120
DRAGAPULT_EX = 121
TEAM_ROCKET_ENERGY = 15
IGNITION_ENERGY = 17
ROCKET_FEATHER_ATTACK = 1285
TAUNT_ATTACK = 1286
PORYGON2_R_COMMAND_ATTACKS = {670}
MURKROW_TEMPT_ATTACK = 652
MURKROW_SECONDARY_ATTACK = 653
MURKROW_ATTACKS = {MURKROW_TEMPT_ATTACK, MURKROW_SECONDARY_ATTACK}
RULE_ONLY_MURKROW_KO_ATTACKS = (TAUNT_ATTACK, MURKROW_SECONDARY_ATTACK)
HOP_PHANTUMP = 878
HOP_PHANTUMP_DODGE_ATTACK = 1266
HOP_PHANTUMP_DODGE_ATTACK_IDS = {HOP_PHANTUMP_DODGE_ATTACK}
ROCKET_SUPPORTER_COST_PRIORITY = {
    1220: 100,  # Team Rocket's Proton / Lance: lowest future value after setup
    1218: 90,  # Team Rocket's Giovanni / Sakaki
    1217: 80,  # Team Rocket's Archer / Apollo
    1219: 70,  # Team Rocket's Petrel / Lambda
    1216: -250,  # Team Rocket's Ariana / Athena: preserve if possible
}
ROCKET_SUPPORTERS = set(ROCKET_SUPPORTER_COST_PRIORITY)
BASIC_SETUP_POKEMON = {MURKROW, PORYGON}
ROCKET_FIELD_POKEMON = {MURKROW, HONCHKROW, PORYGON, PORYGON2, ARTICUNO}
DRAGAPULT_LINE = {DREEPY, DRAKLOAK, DRAGAPULT_EX}
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
DRAW_RESET_SUPPORTERS = {ARCHER, ARIANA}
ABRA = 741
KADABRA = 742
ALAKAZAM = 743
TWM_ABRA = 109
TWM_ALAKAZAM = 245
ALAKAZAM_LINE = {TWM_ABRA, TWM_ALAKAZAM, ABRA, KADABRA, ALAKAZAM}
WEAKNESS_IDS_BY_TYPE = {
    "grass": {
        21, 22, 23, 41, 57, 61, 81, 82, 83, 117, 135, 136, 137, 187, 188, 189,
        225, 227, 228, 229, 248, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291,
        292, 293, 312, 334, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389,
        440, 441, 442, 443, 444, 447, 464, 526, 527, 532, 533, 534, 538, 539, 540, 599,
        600, 601, 603, 604, 607, 608, 609, 610, 611, 614, 615, 616, 617, 618, 642, 643,
        644, 645, 646, 647, 648, 649, 669, 670, 671, 675, 676, 682, 683, 684, 685, 686,
        687, 688, 690, 691, 775, 776, 820, 821, 822, 823, 824, 827, 828, 830, 831, 832,
        881, 882, 887, 888, 889, 892, 893, 894, 895, 896, 897, 973, 975, 976, 985, 986,
        1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1056, 1063,
    },
    "fire": {
        25, 26, 27, 28, 29, 45, 67, 68, 69, 70, 71, 72, 73, 74, 75, 84,
        85, 86, 88, 89, 90, 91, 92, 93, 94, 95, 96, 118, 126, 127, 128, 129,
        142, 143, 146, 147, 148, 149, 150, 165, 166, 167, 168, 169, 170, 177, 178, 190,
        191, 192, 198, 199, 200, 201, 235, 236, 237, 238, 252, 253, 254, 255, 294, 295,
        296, 297, 299, 321, 322, 323, 335, 336, 338, 341, 342, 343, 344, 345, 346, 347,
        348, 349, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 465, 467,
        479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 543, 544, 545, 546, 547,
        557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 619, 620, 621, 622, 623, 624,
        639, 640, 641, 650, 651, 652, 653, 654, 655, 656, 657, 658, 694, 695, 696, 697,
        698, 699, 700, 706, 707, 708, 709, 710, 711, 712, 713, 714, 778, 779, 780, 781,
        782, 783, 784, 785, 786, 787, 835, 836, 837, 838, 839, 840, 850, 851, 852, 853,
        854, 899, 900, 901, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920,
        921, 922, 923, 924, 925, 987, 988, 989, 990, 991, 992, 993, 1011, 1012, 1013, 1014,
        1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1065, 1066, 1067, 1068,
    },
    "water": {
        30, 31, 46, 76, 77, 78, 79, 97, 98, 99, 151, 152, 153, 202, 203, 204,
        205, 239, 256, 257, 258, 259, 319, 320, 324, 325, 326, 350, 351, 352, 353, 354,
        355, 356, 357, 358, 405, 406, 408, 409, 410, 411, 412, 413, 490, 491, 492, 493,
        494, 495, 496, 497, 567, 568, 569, 570, 571, 572, 573, 659, 660, 661, 662, 663,
        664, 665, 666, 715, 716, 717, 718, 719, 788, 789, 790, 791, 792, 793, 794, 795,
        796, 797, 855, 856, 857, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936,
        1025, 1026, 1027,
    },
    "lightning": {
        33, 34, 35, 47, 48, 49, 50, 51, 64, 100, 102, 105, 106, 107, 108, 123,
        155, 157, 158, 159, 172, 173, 174, 180, 181, 182, 197, 206, 207, 208, 240, 241,
        242, 250, 260, 261, 262, 263, 264, 270, 271, 277, 298, 307, 308, 311, 314, 318,
        327, 337, 339, 340, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370,
        407, 414, 415, 416, 417, 420, 421, 422, 425, 456, 457, 458, 463, 466, 476, 477,
        478, 498, 499, 500, 501, 502, 503, 504, 505, 541, 542, 551, 552, 553, 574, 575,
        576, 577, 578, 579, 583, 589, 590, 591, 632, 633, 634, 638, 667, 689, 701, 702,
        720, 721, 724, 725, 726, 727, 728, 757, 773, 774, 798, 799, 803, 804, 805, 826,
        858, 859, 891, 937, 938, 939, 945, 946, 947, 952, 953, 983, 984, 1003, 1004, 1005,
        1007, 1008, 1023, 1024, 1028, 1029, 1030, 1031, 1034, 1062, 1064, 1075,
    },
    "psychic": {
        40, 58, 113, 114, 115, 116, 224, 226, 333, 437, 438, 439, 445, 446, 528, 529,
        530, 531, 602, 605, 606, 672, 673, 674, 677, 678, 679, 680, 681, 819, 883, 884,
        885, 886, 890, 972, 974, 977, 978, 979, 1055,
    },
    "fighting": {
        24, 36, 37, 43, 44, 52, 59, 60, 65, 66, 124, 125, 130, 138, 139, 140,
        141, 145, 160, 161, 175, 176, 210, 211, 212, 213, 230, 232, 233, 234, 244, 249,
        251, 265, 266, 267, 268, 269, 304, 305, 306, 309, 310, 317, 328, 329, 371, 372,
        373, 374, 375, 376, 377, 390, 391, 392, 426, 427, 428, 448, 449, 450, 451, 452,
        453, 454, 455, 459, 460, 461, 462, 468, 469, 470, 471, 472, 473, 474, 475, 510,
        511, 512, 513, 514, 515, 535, 536, 537, 554, 555, 556, 584, 585, 586, 587, 588,
        612, 613, 626, 627, 628, 629, 630, 631, 692, 693, 703, 704, 705, 732, 733, 734,
        735, 736, 737, 738, 739, 740, 756, 758, 759, 760, 761, 770, 771, 772, 777, 806,
        807, 808, 809, 810, 811, 814, 829, 833, 834, 841, 842, 843, 844, 845, 846, 847,
        848, 849, 868, 869, 870, 871, 872, 898, 948, 949, 950, 951, 954, 955, 956, 957,
        980, 981, 982, 996, 997, 998, 999, 1000, 1001, 1002, 1006, 1009, 1010, 1035, 1036, 1037,
        1038, 1057, 1058, 1059, 1060, 1061, 1069, 1070, 1071, 1072, 1073, 1074, 1076,
    },
    "darkness": {
        38, 54, 55, 80, 109, 111, 112, 131, 132, 133, 162, 163, 183, 184, 185, 186,
        215, 216, 217, 218, 219, 220, 221, 223, 245, 246, 273, 274, 275, 276, 332, 429,
        430, 431, 432, 433, 434, 435, 436, 516, 517, 518, 519, 520, 521, 522, 523, 524,
        525, 592, 593, 594, 595, 596, 597, 598, 635, 636, 668, 741, 742, 743, 744, 745,
        746, 747, 748, 749, 750, 752, 753, 764, 765, 812, 813, 817, 818, 873, 874, 875,
        876, 877, 878, 879, 880, 963, 964, 969, 971, 1042, 1043,
    },
    "metal": {
        32, 39, 53, 56, 101, 103, 104, 110, 134, 154, 156, 164, 179, 209, 214, 222,
        243, 247, 272, 278, 279, 280, 315, 316, 330, 331, 418, 419, 423, 424, 506, 507,
        508, 509, 580, 581, 582, 637, 722, 723, 729, 730, 731, 751, 762, 763, 766, 767,
        768, 769, 800, 801, 802, 815, 816, 825, 860, 861, 862, 863, 864, 865, 866, 867,
        940, 941, 942, 943, 944, 958, 959, 960, 961, 962, 965, 966, 967, 968, 970, 1032,
        1033, 1039, 1040, 1041, 1044, 1045,
    },
}

RESISTANCE_IDS_BY_TYPE = {
    "grass": {
        84, 85, 86, 118, 142, 143, 165, 166, 167, 168, 169, 170, 190, 191, 192, 294,
        295, 296, 297, 299, 335, 336, 465, 467, 543, 544, 545, 546, 547, 619, 620, 621,
        622, 623, 624, 639, 640, 641, 694, 695, 696, 697, 698, 699, 700, 835, 836, 837,
        838, 839, 840, 899, 900, 901, 987, 988, 989, 990, 991, 992, 993, 1065, 1066, 1067,
        1068,
    },
    "fighting": {
        38, 54, 55, 64, 80, 109, 111, 112, 123, 131, 132, 133, 162, 163, 172, 173,
        174, 183, 184, 185, 186, 197, 215, 216, 217, 218, 219, 220, 221, 223, 245, 246,
        250, 270, 271, 273, 274, 275, 276, 277, 298, 307, 308, 311, 318, 332, 337, 339,
        340, 407, 414, 425, 429, 430, 431, 432, 433, 434, 435, 436, 456, 457, 458, 463,
        466, 476, 477, 478, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 541, 542,
        551, 552, 553, 578, 579, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 632,
        633, 634, 635, 636, 638, 668, 689, 701, 702, 741, 742, 743, 744, 745, 746, 747,
        748, 749, 750, 752, 753, 757, 764, 765, 773, 774, 812, 813, 817, 818, 826, 873,
        874, 875, 876, 877, 878, 879, 880, 891, 952, 953, 963, 964, 969, 971, 983, 984,
        1003, 1004, 1005, 1007, 1008, 1023, 1024, 1042, 1043, 1062, 1064, 1075,
    },
}

DARK_WEAK_TO_IDS = WEAKNESS_IDS_BY_TYPE["darkness"]
DARK_WEAK_NAME_HINTS = (
    "abra",
    "kadabra",
    "alakazam",
    "mewtwo",
)
# Prize count table generated with the same card-id discipline as weakness data:
# Pokemon cards with explicit ex/EX prize rules are listed here by competition id.
PRIZE_COUNT_IDS_BY_VALUE = {
    2: {
    24, 29, 30, 37, 40, 44, 46, 52, 63, 75, 79, 80, 83, 84, 96, 99,
    107, 108, 117, 121, 125, 130, 138, 139, 140, 141, 150, 153, 154, 161, 176, 179,
    184, 189, 190, 193, 198, 205, 207, 210, 223, 229, 231, 232, 236, 239, 241, 243,
    244, 246, 248, 249, 259, 269, 272, 283, 293, 299, 302, 306, 313, 316, 320, 326,
    328, 329, 331, 336, 337, 340, 357, 369, 372, 381, 389, 404, 407, 424, 431, 447,
    455, 458, 471, 481, 509, 515, 525, 527, 547, 561, 573, 583, 598, 618, 631, 641,
    648, 795, 806, 813, 835, 911, 944, 951, 954, 957, 962, 968, 969, 975, 979, 984,
    988, 990, 993, 997, 1002, 1022, 1026, 1062, 1071,
    },
    3: {
    652, 662, 678, 687, 695, 723, 737, 747, 754, 756, 766, 772, 781, 790, 828, 849,
    861, 868, 886, 896, 904, 919, 928, 932, 939, 1006, 1031, 1040, 1056, 1064,
    },
}
PRIZE_COUNT_BY_CARD_ID = {
    card_id: prize_count
    for prize_count, card_ids in PRIZE_COUNT_IDS_BY_VALUE.items()
    for card_id in card_ids
}
TWO_PRIZE_POKEMON_IDS = PRIZE_COUNT_IDS_BY_VALUE[2]
THREE_PRIZE_POKEMON_IDS = PRIZE_COUNT_IDS_BY_VALUE[3]


def load_deck():
    candidates = []
    module_file = globals().get("__file__")
    if module_file:
        candidates.append(Path(module_file).resolve().parent / "deck.csv")
    candidates.extend((
        Path("deck.csv"),
        Path("/kaggle_simulations/agent/deck.csv"),
    ))
    for search_entry in reversed(sys.path):
        if search_entry:
            candidates.append(Path(search_entry) / "deck.csv")

    seen = set()
    for deck_path in candidates:
        try:
            resolved = deck_path.resolve()
        except Exception:
            resolved = deck_path
        if resolved in seen:
            continue
        seen.add(resolved)
        if deck_path.exists():
            deck = [int(line.strip()) for line in deck_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(deck) != 60:
                raise ValueError(f"deck.csv must contain exactly 60 cards, got {len(deck)}")
            return deck
    raise FileNotFoundError("deck.csv was not found")


MY_DECK = load_deck()
PUBLIC_DECK_MEMORY = {}
HOP_DODGE_MEMORY = {}
OBSERVED_ATTACK_MEMORY = {}
OBSERVED_ATTACK_LOG_KEYS = set()


def _reset_public_knowledge():
    PUBLIC_DECK_MEMORY.clear()
    HOP_DODGE_MEMORY.clear()
    OBSERVED_ATTACK_MEMORY.clear()
    OBSERVED_ATTACK_LOG_KEYS.clear()


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
    if identifier == NIGHT_STRETCHER:
        return _night_stretcher_immediate_compression_score(observation) >= 75_000
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


def _normalize_attack_type(type_name):
    raw = str(type_name or "").strip().lower().replace(" ", "-")
    return {
        "dark": "darkness",
        "electric": "lightning",
        "steel": "metal",
        "none": "colorless",
    }.get(raw, raw)


def _pokemon_has_weakness_to(pokemon, attack_type):
    defender_id = _card_id(pokemon)
    normalized_type = _normalize_attack_type(attack_type)
    if defender_id in WEAKNESS_IDS_BY_TYPE.get(normalized_type, set()):
        return True
    if normalized_type == "darkness":
        defender_name = str(_read(pokemon, "name", "")).lower()
        return any(hint in defender_name for hint in DARK_WEAK_NAME_HINTS)
    return False


def _pokemon_has_resistance_to(pokemon, attack_type):
    defender_id = _card_id(pokemon)
    normalized_type = _normalize_attack_type(attack_type)
    return defender_id in RESISTANCE_IDS_BY_TYPE.get(normalized_type, set())


def _damage_after_type_modifier(base_damage, defender, attack_type):
    damage = max(0, int(base_damage or 0))
    if _pokemon_has_weakness_to(defender, attack_type):
        damage *= 2
    if _pokemon_has_resistance_to(defender, attack_type):
        damage = max(0, damage - 30)
    return damage


def _rocket_feather_damage_per_supporter(opponent_active):
    return _damage_after_type_modifier(60, opponent_active, "darkness")


def _murkrow_taunt_damage(opponent_active):
    return _damage_after_type_modifier(30, opponent_active, "darkness")


def _active_murkrow_can_evolve_to_honchkrow(observation):
    current, _, player = _current_player(observation)
    active = _top_card(_read(player, "active", []))
    if _card_id(active) != MURKROW:
        return False
    select = _read(observation, "select", {})
    for option in _select_options(select):
        if _option_type(option) not in (9, "evolve"):
            continue
        if _card_id(_card_from_option(observation, option)) != HONCHKROW:
            continue
        target = _target_card_from_option(observation, option)
        target_is_active = _area_code(_read(option, "inPlayArea"), -1) == 4
        if target_is_active and (_card_id(target) in (None, MURKROW)):
            return True
        if target is None and target_is_active:
            return True
    return False


def _deck_basic_setup_count(player):
    return _deck_card_count_for_policy(player, lambda card_id: card_id in BASIC_SETUP_POKEMON)


def _tempt_supporter_target_score(player, identifier):
    if identifier not in ROCKET_SUPPORTERS:
        return -100_000

    hand_ids = [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]
    if ARIANA not in hand_ids:
        if identifier == ARIANA:
            return 95_000
        return -20_000

    if _deck_basic_setup_count(player) >= 3 and PROTON not in hand_ids:
        if identifier == PROTON:
            return 88_000
        return -12_000

    missing_priority = (
        (ARCHER, 78_000),
        (PETREL, 68_000),
        (GIOVANNI, 58_000),
    )
    for supporter_id, score in missing_priority:
        if supporter_id not in hand_ids:
            return score if identifier == supporter_id else -8_000

    fallback_scores = {
        ARCHER: 19_000,
        PETREL: 16_000,
        GIOVANNI: 13_000,
        ARIANA: 8_000,
        PROTON: 2_000,
    }
    return fallback_scores.get(identifier, -8_000)


def _night_stretcher_target_score(observation, card):
    identifier = _card_id(card)
    if identifier is None:
        return -100_000

    current, your_index, player = _current_player(observation)
    field_top_ids = [_card_id(field_card) for field_card in _field_top_cards(player)]
    hand_ids = [_card_id(hand_card) for hand_card in _iter_cards(_read(player, "hand", []))]
    discard_supporters = _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    bench_count = _bench_top_count(player)
    field_count = len(field_top_ids)
    free_bench = bench_count < _bench_limit(player)
    seed_out_risk = field_count <= 1 and free_bench
    thin_board_basic_risk = field_count <= 2 and free_bench
    has_murkrow_source = MURKROW in field_top_ids or MURKROW in hand_ids
    has_porygon_source = PORYGON in field_top_ids or PORYGON in hand_ids
    has_honchkrow_ready = HONCHKROW in field_top_ids or HONCHKROW in hand_ids
    has_porygon2_ready = PORYGON2 in field_top_ids or PORYGON2 in hand_ids

    if _alakazam_articuno_recovery_needed(observation) and identifier == ARTICUNO:
        return 900_000

    alakazam_basic_score = _alakazam_basic_target_score(observation, identifier)
    if alakazam_basic_score is not None:
        return alakazam_basic_score

    dragapult_basic_score = _dragapult_basic_target_score(observation, identifier)
    if dragapult_basic_score is not None:
        return dragapult_basic_score
    if _dragapult_matchup_active(observation) and identifier in (HONCHKROW, PORYGON2):
        if not _dragapult_evolution_attack_ready(observation, identifier):
            return -95_000

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
        if discard_supporters >= _porygon2_late_trash_threshold() and has_porygon_source:
            return 240_000
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


def _night_stretcher_immediate_compression_score(observation):
    _, _, player = _current_player(observation)
    field_top_ids = [_card_id(field_card) for field_card in _field_top_cards(player)]
    free_bench = _bench_top_count(player) < _bench_limit(player)
    best = -100_000

    for zone in ("discard", "trash"):
        for card in _iter_cards(_read(player, zone, [])):
            identifier = _card_id(card)
            if identifier in BASIC_SETUP_POKEMON and free_bench:
                score = 176_000 if identifier == MURKROW else 142_000
            elif identifier == HONCHKROW and MURKROW in field_top_ids:
                score = 168_000
            elif identifier == PORYGON2 and PORYGON in field_top_ids and _porygon_development_allowed(player):
                score = 154_000
            else:
                continue
            if score > best:
                best = score

    return best


def _poke_pad_target_score(observation, card):
    identifier = _card_id(card)
    if identifier is None:
        return -100_000

    _, _, player = _current_player(observation)
    current = _read(observation, "current", {})
    field_top_ids = [_card_id(field_card) for field_card in _field_top_cards(player)]
    hand_ids = [_card_id(hand_card) for hand_card in _iter_cards(_read(player, "hand", []))]
    deck_ids = _deck_card_ids_for_policy(player)
    discard_supporters = _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    bench_count = _bench_top_count(player)
    field_count = len(field_top_ids)
    free_bench = bench_count < _bench_limit(player)
    turn_number = _safe_int(_read(current, "turn"), 0)
    seed_out_risk = field_count <= 1 and free_bench
    thin_board_basic_risk = field_count <= 2 and free_bench
    has_murkrow_source = MURKROW in field_top_ids or MURKROW in hand_ids
    has_porygon_source = PORYGON in field_top_ids or PORYGON in hand_ids
    has_honchkrow_ready = HONCHKROW in field_top_ids or HONCHKROW in hand_ids
    has_porygon2_ready = PORYGON2 in field_top_ids or PORYGON2 in hand_ids
    honchkrow_pad_bridge = _honchkrow_pad_bridge_available(player, hand_ids, deck_ids)

    alakazam_basic_score = _alakazam_basic_target_score(observation, identifier)
    if alakazam_basic_score is not None:
        return alakazam_basic_score

    dragapult_basic_score = _dragapult_basic_target_score(observation, identifier)
    if dragapult_basic_score is not None:
        return dragapult_basic_score
    if _dragapult_matchup_active(observation) and identifier in (HONCHKROW, PORYGON2):
        if not _dragapult_evolution_attack_ready(observation, identifier):
            return -95_000

    if _is_basic_setup_pokemon(identifier):
        if identifier == MURKROW and honchkrow_pad_bridge and turn_number <= 1:
            return 86_000
        if seed_out_risk:
            return 300_000 + (24_000 if identifier == MURKROW else 12_000)
        if thin_board_basic_risk:
            return 245_000 + (24_000 if identifier == MURKROW else 12_000)
        score = 118_000 if identifier == MURKROW else 92_000
        if free_bench:
            score += 34_000
        if identifier == MURKROW and not has_honchkrow_ready and not honchkrow_pad_bridge:
            score += 16_000
        if identifier == PORYGON and not has_porygon2_ready:
            score += 10_000
        return score

    if seed_out_risk or thin_board_basic_risk:
        if identifier == HONCHKROW and honchkrow_pad_bridge and turn_number <= 1:
            return 275_000
        return -85_000
    if identifier == HONCHKROW:
        if honchkrow_pad_bridge and turn_number <= 1:
            return 275_000
        return 180_000 if has_murkrow_source else 22_000
    if identifier == PORYGON2:
        if not _porygon_development_allowed(player):
            return 12_000
        if discard_supporters >= _porygon2_late_trash_threshold() and has_porygon_source:
            return 240_000
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
        for identifier in _deck_card_ids_for_policy(player):
            if identifier in ROCKET_FIELD_POKEMON:
                targets.append({"id": identifier})
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


def _card_serial(card):
    value = _read(card, "serial")
    if value is None:
        return None
    return _safe_int(value, None)


def _compact_public_card(card):
    identifier = _card_id(card)
    if identifier is None:
        return None
    compact = {"id": identifier}
    serial = _card_serial(card)
    if serial is not None:
        compact["serial"] = serial
    player_index = _read(card, "playerIndex")
    if player_index is not None:
        compact["playerIndex"] = _safe_int(player_index, 0)
    name = _read(card, "name")
    if name:
        compact["name"] = name
    return compact


def _physical_cards_from_in_play(card):
    top = _top_card(card)
    if top is None:
        return
    yield top
    for key in ("preEvolution", "energyCards", "tools"):
        for nested in _iter_cards(_read(top, key, [])):
            yield nested


def _visible_non_deck_cards(player):
    for zone in ("hand", "discard", "trash", "prize", "prizes"):
        for card in _iter_cards(_read(player, zone, [])):
            yield card
    for zone in ("active", "bench"):
        cards = _read(player, zone, [])
        if not isinstance(cards, list):
            continue
        for slot in cards:
            for card in _iter_cards(slot):
                yield from _physical_cards_from_in_play(card)


def _visible_non_deck_serials(player):
    serials = set()
    for card in _visible_non_deck_cards(player):
        serial = _card_serial(card)
        if serial is not None:
            serials.add(serial)
    return serials


def _player_index_hint(player):
    value = _read(player, "playerIndex")
    if value is not None:
        return _safe_int(value, None)
    for card in _visible_non_deck_cards(player):
        value = _read(card, "playerIndex")
        if value is not None:
            return _safe_int(value, None)
    return None


def _remember_public_information(observation):
    select = _select_payload(observation)
    deck = _read(select, "deck", None)
    if not isinstance(deck, list):
        return

    cards = []
    player_indexes = []
    for card in deck:
        compact = _compact_public_card(_top_card(card))
        if compact is None:
            continue
        cards.append(compact)
        player_index = compact.get("playerIndex")
        if player_index is not None:
            player_indexes.append(player_index)
    if not cards and deck:
        return

    current = _read(observation, "current", {})
    if player_indexes:
        player_index = max(set(player_indexes), key=player_indexes.count)
    else:
        player_index = _safe_int(_read(current, "yourIndex"), 0)

    player = _player_state(current, player_index)
    deck_count = _deck_count(player)
    if deck_count > 0 and len(cards) != deck_count:
        return

    PUBLIC_DECK_MEMORY[player_index] = cards


def _known_public_deck_cards(player):
    player_index = _player_index_hint(player)
    if player_index is None:
        return None
    cards = PUBLIC_DECK_MEMORY.get(player_index)
    if not isinstance(cards, list):
        return None

    visible_serials = _visible_non_deck_serials(player)
    if visible_serials:
        cards = [
            card for card in cards
            if _card_serial(card) is None or _card_serial(card) not in visible_serials
        ]

    deck_count = _deck_count(player)
    if deck_count > 0 and len(cards) != deck_count:
        return None
    return cards


def _estimated_remaining_deck_ids(player):
    remaining = list(MY_DECK)
    for card in _visible_non_deck_cards(player):
        identifier = _card_id(card)
        if identifier in remaining:
            remaining.remove(identifier)
    return remaining


def _deck_card_ids_for_policy(player):
    known = _known_public_deck_cards(player)
    if known is not None:
        return [_card_id(card) for card in known if _card_id(card) is not None]

    deck = _read(player, "deck", [])
    visible_ids = [_card_id(card) for card in _iter_cards(deck)]
    visible_ids = [identifier for identifier in visible_ids if identifier is not None]
    if visible_ids:
        return visible_ids

    return _estimated_remaining_deck_ids(player)


def _deck_card_count_for_policy(player, predicate):
    return sum(1 for identifier in _deck_card_ids_for_policy(player) if predicate(identifier))


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


def _dragapult_matchup_active(observation):
    current = _read(observation, "current", {})
    your_index = _safe_int(_read(current, "yourIndex"), 0)
    opponent = _player_state(current, 1 - your_index)
    return any(_card_id(card) in DRAGAPULT_LINE for card in _field_top_cards(opponent))


def _dragapult_basic_priority_ids(player):
    field_top_ids = [_card_id(card) for card in _field_top_cards(player)]
    wanted = []
    if ARTICUNO not in field_top_ids:
        wanted.append(ARTICUNO)
    if field_top_ids.count(MURKROW) < 2:
        wanted.append(MURKROW)
    wanted.append(PORYGON)
    return tuple(wanted)


def _dragapult_articuno_accessible(player):
    field_top_ids = [_card_id(card) for card in _field_top_cards(player)]
    if ARTICUNO in field_top_ids:
        return True

    hand_ids = [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]
    if ARTICUNO in hand_ids:
        return True

    deck_ids = _deck_card_ids_for_policy(player)
    if ARTICUNO in deck_ids:
        return True

    discard_ids = [_card_id(card) for card in _iter_cards(_read(player, "discard", []))]
    if ARTICUNO in discard_ids and (NIGHT_STRETCHER in hand_ids or NIGHT_STRETCHER in deck_ids):
        return True
    return False


def _dragapult_articuno_search_needed(observation):
    if not _dragapult_matchup_active(observation):
        return False

    _, _, player = _current_player(observation)
    if _bench_top_count(player) >= _bench_limit(player):
        return False

    field_top_ids = [_card_id(card) for card in _field_top_cards(player)]
    if ARTICUNO in field_top_ids:
        return False

    hand_ids = [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]
    if ARTICUNO in hand_ids:
        return False

    return ARTICUNO in _deck_card_ids_for_policy(player)


def _dragapult_articuno_access_action_ready(observation, player):
    hand_ids = [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]
    if ARTICUNO in hand_ids:
        return True

    current = _read(observation, "current", {})
    supporter_played = _supporter_played_this_turn(current, player)
    deck_ids = _deck_card_ids_for_policy(player)
    if ARTICUNO in deck_ids:
        if POKE_PAD in hand_ids:
            return True
        if not supporter_played and PROTON in hand_ids:
            return True
        if any(card_id in hand_ids for card_id in (TEAM_ROCKET_TRANSCEIVER, POKEGEAR, ROTO_STICK)):
            return True

    discard_ids = [_card_id(card) for card in _iter_cards(_read(player, "discard", []))]
    return ARTICUNO in discard_ids and NIGHT_STRETCHER in hand_ids


def _dragapult_should_hold_porygon_for_articuno(observation, identifier):
    if identifier != PORYGON or not _dragapult_matchup_active(observation):
        return False

    _, _, player = _current_player(observation)
    field_top_ids = [_card_id(card) for card in _field_top_cards(player)]
    if ARTICUNO in field_top_ids:
        return False
    if not _dragapult_articuno_accessible(player):
        return False

    porygon_lines = field_top_ids.count(PORYGON) + field_top_ids.count(PORYGON2)
    free_bench_slots = max(0, _bench_limit(player) - _bench_top_count(player))
    if _dragapult_articuno_access_action_ready(observation, player):
        return True
    return porygon_lines > 0 or free_bench_slots <= 1


def _dragapult_forbidden_basic_play_option(observation, option):
    if _option_type(option) not in (7, "play"):
        return False
    return _dragapult_should_hold_porygon_for_articuno(observation, _rule_option_id(observation, option))


def _dragapult_basic_target_score(observation, identifier):
    if not _dragapult_matchup_active(observation):
        return None
    _, _, player = _current_player(observation)
    if _bench_top_count(player) >= _bench_limit(player):
        return -85_000 if identifier in (MURKROW, ARTICUNO, PORYGON) else None
    field_top_ids = [_card_id(card) for card in _field_top_cards(player)]
    murkrow_count = field_top_ids.count(MURKROW)
    has_articuno = ARTICUNO in field_top_ids
    if identifier == MURKROW:
        if murkrow_count < 2:
            return 300_000 - murkrow_count * 12_000 if not has_articuno else 430_000 - murkrow_count * 12_000
        return 48_000
    if identifier == ARTICUNO:
        if has_articuno:
            return -70_000
        return 540_000
    if identifier == PORYGON:
        if _dragapult_should_hold_porygon_for_articuno(observation, identifier):
            return -120_000
        return 300_000 if murkrow_count >= 2 and has_articuno else 155_000
    return None


def _player_zone_has_card_id(player, zones, identifier):
    for zone in zones:
        for card in _iter_cards(_read(player, zone, [])):
            if _card_id(card) == identifier:
                return True
    return False


def _alakazam_matchup_active(observation):
    current = _read(observation, "current", {})
    your_index = _safe_int(_read(current, "yourIndex"), 0)
    opponent = _player_state(current, 1 - your_index)
    return any(_card_id(card) in ALAKAZAM_LINE for card in _field_top_cards(opponent))


def _alakazam_articuno_confirmed_unavailable(observation):
    if not _alakazam_matchup_active(observation):
        return False

    _, _, player = _current_player(observation)
    if _player_zone_has_card_id(player, ("active", "bench", "hand"), ARTICUNO):
        return False
    if _player_zone_has_card_id(player, ("prize", "prizes"), ARTICUNO):
        return True

    if _player_zone_has_card_id(player, ("discard", "trash"), ARTICUNO):
        hand_ids = _rule_hand_ids(player)
        deck_ids = _deck_card_ids_for_policy(player)
        return NIGHT_STRETCHER not in hand_ids and NIGHT_STRETCHER not in deck_ids

    known = _known_public_deck_cards(player)
    if known is not None:
        return ARTICUNO not in [_card_id(card) for card in known]
    return False


def _alakazam_resist_veil_plan_active(observation):
    return _alakazam_matchup_active(observation) and not _alakazam_articuno_confirmed_unavailable(observation)


def _alakazam_side_race_finish_ready(observation):
    if not _alakazam_resist_veil_plan_active(observation):
        return False

    _, _, player = _current_player(observation)
    remaining_prizes = _prize_cards_remaining(player)
    if remaining_prizes <= 0:
        return True

    hand_ids = _rule_hand_ids(player)
    deck_ids = _deck_card_ids_for_policy(player)
    available_honchkrow = hand_ids.count(HONCHKROW) + deck_ids.count(HONCHKROW)
    prepared_murkrow = 0
    ready_honchkrow = 0
    hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)

    for card in _field_top_cards(player):
        identifier = _card_id(card)
        energy_ids = _attached_energy_card_ids(card)
        if identifier == MURKROW and TEAM_ROCKET_ENERGY in energy_ids:
            prepared_murkrow += 1
        elif identifier == HONCHKROW and _attached_energy_cards(card) > 0 and hand_supporters > 0:
            ready_honchkrow += 1

    prepared_attackers = ready_honchkrow + min(prepared_murkrow, available_honchkrow)
    return prepared_attackers >= remaining_prizes


def _alakazam_prize_loss_pressure(observation):
    if not _alakazam_resist_veil_plan_active(observation):
        return False
    current, your_index, _ = _current_player(observation)
    opponent = _player_state(current, 1 - your_index)
    return _prize_cards_remaining(opponent) <= 2


def _alakazam_low_deck_release_ready(observation):
    if not _alakazam_resist_veil_plan_active(observation):
        return False

    _, _, player = _current_player(observation)
    remaining_prizes = _prize_cards_remaining(player)
    endgame_prize_threshold = _policy_rule_number("alakazamLockPlan", "endgameReleasePrizeThreshold", 3)
    if remaining_prizes > endgame_prize_threshold and not _alakazam_prize_loss_pressure(observation):
        return False

    hand_ids = _rule_hand_ids(player)
    deck_ids = _deck_card_ids_for_policy(player)
    field_top_ids = [_card_id(card) for card in _field_top_cards(player)]
    has_honchkrow_path = HONCHKROW in field_top_ids or (
        MURKROW in field_top_ids and (HONCHKROW in hand_ids or HONCHKROW in deck_ids)
    )
    has_porygon2_path = PORYGON2 in field_top_ids or (
        PORYGON in field_top_ids and (PORYGON2 in hand_ids or PORYGON2 in deck_ids)
    )
    return has_honchkrow_path or has_porygon2_path


def _alakazam_lock_strategy_active(observation):
    return (
        _alakazam_resist_veil_plan_active(observation)
        and not _alakazam_side_race_finish_ready(observation)
        and not _alakazam_low_deck_release_ready(observation)
    )


def _alakazam_taunt_stable(observation, options=None):
    if not _alakazam_taunt_lock_ready(observation):
        return False
    if options is not None and _rule_find_attack_option(options, RULE_ONLY_MURKROW_KO_ATTACKS) is None:
        return False
    return True


def _alakazam_taunt_lock_ready(observation):
    if not _alakazam_lock_strategy_active(observation):
        return False
    _, _, player = _current_player(observation)
    if _needs_seed_out_bench_guard(player):
        return False
    field_top_ids = [_card_id(card) for card in _field_top_cards(player)]
    if ARTICUNO not in field_top_ids and not _alakazam_articuno_confirmed_unavailable(observation):
        return False
    active = _top_card(_read(player, "active", []))
    return _card_id(active) == MURKROW


def _alakazam_lock_deck_preservation_active(observation):
    if not _alakazam_lock_strategy_active(observation):
        return False
    _, _, player = _current_player(observation)
    if _needs_seed_out_bench_guard(player):
        return False
    field_top_ids = [_card_id(card) for card in _field_top_cards(player)]
    return ARTICUNO in field_top_ids and field_top_ids.count(MURKROW) >= 2


def _alakazam_articuno_recovery_needed(observation):
    if not _alakazam_lock_strategy_active(observation):
        return False
    _, _, player = _current_player(observation)
    field_top_ids = [_card_id(card) for card in _field_top_cards(player)]
    if ARTICUNO in field_top_ids:
        return False
    if _bench_top_count(player) >= _bench_limit(player) and ARTICUNO not in _rule_hand_ids(player):
        return False
    return _player_zone_has_card_id(player, ("discard", "trash"), ARTICUNO)


def _alakazam_articuno_petrel_recovery_needed(observation):
    if not _alakazam_articuno_recovery_needed(observation):
        return False
    current, _, player = _current_player(observation)
    if _supporter_played_this_turn(current, player):
        return False
    hand_ids = _rule_hand_ids(player)
    deck_ids = _deck_card_ids_for_policy(player)
    return NIGHT_STRETCHER not in hand_ids and NIGHT_STRETCHER in deck_ids


def _alakazam_porygon_active_escape_needed(observation):
    if not _alakazam_lock_strategy_active(observation):
        return False
    _, _, player = _current_player(observation)
    active = _top_card(_read(player, "active", []))
    if _card_id(active) != PORYGON:
        return False
    if _needs_seed_out_bench_guard(player):
        return False
    field_top_ids = [_card_id(card) for card in _field_top_cards(player)]
    return ARTICUNO in field_top_ids and MURKROW in field_top_ids


def _alakazam_basic_priority_ids(observation):
    _, _, player = _current_player(observation)
    field_top_ids = [_card_id(card) for card in _field_top_cards(player)]
    if _alakazam_taunt_stable(observation):
        return ()
    wanted = []
    if ARTICUNO not in field_top_ids:
        wanted.append(ARTICUNO)
    target_murkrow = max(2, min(3, _prize_cards_remaining(player)))
    if field_top_ids.count(MURKROW) < target_murkrow:
        wanted.append(MURKROW)
    if _alakazam_lock_strategy_active(observation):
        wanted.append(PORYGON)
    else:
        wanted.extend((MURKROW, HONCHKROW, PORYGON, PORYGON2))
    return tuple(wanted)


def _alakazam_basic_target_score(observation, identifier):
    if not _alakazam_resist_veil_plan_active(observation):
        return None
    if _alakazam_taunt_stable(observation) and identifier in (MURKROW, PORYGON, HONCHKROW, PORYGON2):
        return -95_000
    _, _, player = _current_player(observation)
    field_top_ids = [_card_id(card) for card in _field_top_cards(player)]
    if _bench_top_count(player) >= _bench_limit(player):
        return -90_000 if identifier in (ARTICUNO, MURKROW, PORYGON) else None

    lock_active = _alakazam_lock_strategy_active(observation)
    murkrow_count = field_top_ids.count(MURKROW)
    target_murkrow = max(2, min(3, _prize_cards_remaining(player)))
    if identifier == ARTICUNO:
        return -80_000 if ARTICUNO in field_top_ids else 540_000
    if identifier == MURKROW:
        if murkrow_count < target_murkrow:
            return 470_000 - murkrow_count * 14_000
        return 74_000
    if identifier == PORYGON:
        return 42_000 if lock_active else 105_000
    if lock_active and identifier in (HONCHKROW, PORYGON2):
        return -120_000
    return None


def _alakazam_forbidden_evolution_option(observation, option):
    evolution_id = _rule_option_id(observation, option)
    if evolution_id not in (HONCHKROW, PORYGON2):
        return False
    return _alakazam_lock_strategy_active(observation)


def _alakazam_promotion_score(observation, option):
    if not _alakazam_resist_veil_plan_active(observation):
        return None
    card = _card_from_option(observation, option)
    identifier = _card_id(card)
    if identifier == ARTICUNO:
        return -1_000_000
    energy_count = _attached_energy_cards(card)
    damaged = _card_is_damaged(card)
    if identifier == MURKROW and energy_count > 0 and damaged:
        return 640_000
    if identifier == MURKROW and energy_count > 0:
        return 585_000
    if identifier == MURKROW and damaged:
        return 410_000
    if identifier == MURKROW:
        return 320_000
    if identifier == PORYGON and damaged:
        return 210_000
    if identifier == PORYGON:
        return 160_000
    if identifier in (HONCHKROW, PORYGON2):
        return 80_000 if _alakazam_side_race_finish_ready(observation) else -95_000
    return -55_000


def _card_is_damaged(card):
    if card is None:
        return False
    damage = _safe_int(_read(card, "damage", _read(card, "damageCounters", _read(card, "damageCounter"))), 0)
    if damage > 0:
        return True
    max_hp = _safe_int(_read(card, "maxHp", _read(card, "maxHP")), 0)
    remaining_hp = _remaining_hp(card)
    return max_hp > 0 and 0 < remaining_hp < max_hp


def _dragapult_evolution_attack_ready(observation, evolution_id, option=None, target_card=None):
    if not _dragapult_matchup_active(observation):
        return True
    if evolution_id not in (HONCHKROW, PORYGON2):
        return True

    current, _, player = _current_player(observation)
    target = target_card
    target_is_active = False
    if option is not None:
        target = _target_card_from_option(observation, option)
        target_is_active = _area_code(_read(option, "inPlayArea"), -1) == 4
    if target is None:
        target = _top_card(_read(player, "active", []))
        target_is_active = True
    if not target_is_active:
        return False

    target_id = _card_id(target)
    hand_ids = _rule_hand_ids(player)
    energy_used = _energy_attached_this_turn(current, player)
    attached = _attached_energy_cards(target) > 0
    hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)

    if evolution_id == HONCHKROW:
        if target_id != MURKROW:
            return False
        energy_access = attached or (not energy_used and any(card_id in hand_ids for card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY)))
        return bool(energy_access and hand_supporters > 0)

    if evolution_id == PORYGON2:
        if target_id != PORYGON:
            return False
        energy_access = attached or (not energy_used and IGNITION_ENERGY in hand_ids)
        discard_supporters = _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS)
        return bool(energy_access and (discard_supporters > 0 or _rule_r_command_pressure_is_worth_ignition(observation)))

    return True


def _dragapult_petrel_poke_pad_attack_bridge_needed(observation):
    if not _dragapult_matchup_active(observation):
        return False

    current, _, player = _current_player(observation)
    hand_ids = _rule_hand_ids(player)
    deck_ids = _deck_card_ids_for_policy(player)
    if PETREL not in hand_ids or POKE_PAD not in deck_ids:
        return False

    active = _top_card(_read(player, "active", []))
    active_id = _card_id(active)
    energy_used = _energy_attached_this_turn(current, player)
    attached = _attached_energy_cards(active) > 0

    if active_id == MURKROW and HONCHKROW in deck_ids and HONCHKROW not in hand_ids:
        energy_access = attached or (
            not energy_used
            and any(card_id in hand_ids for card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY))
        )
        other_hand_supporters = sum(
            1 for card_id in hand_ids if card_id in ROCKET_SUPPORTERS and card_id != PETREL
        )
        return bool(energy_access and other_hand_supporters > 0)

    if active_id == PORYGON and PORYGON2 in deck_ids and PORYGON2 not in hand_ids:
        energy_access = attached or (not energy_used and IGNITION_ENERGY in hand_ids)
        return bool(energy_access and _rule_r_command_pressure_is_worth_ignition(observation))

    return False


def _alakazam_petrel_poke_pad_bridge_needed(observation):
    if not _alakazam_resist_veil_plan_active(observation):
        return False
    if _alakazam_taunt_stable(observation):
        return False
    _, _, player = _current_player(observation)
    deck_ids = _deck_card_ids_for_policy(player)
    if POKE_PAD not in deck_ids:
        return False
    field_top_ids = [_card_id(card) for card in _field_top_cards(player)]
    if ARTICUNO not in field_top_ids and ARTICUNO in deck_ids:
        return True
    if _alakazam_lock_strategy_active(observation):
        target_murkrow = max(2, min(3, _prize_cards_remaining(player)))
        return field_top_ids.count(MURKROW) < target_murkrow and MURKROW in deck_ids
    return False


def _dragapult_forbidden_evolution_option(observation, option):
    evolution_id = _rule_option_id(observation, option)
    if evolution_id not in (HONCHKROW, PORYGON2):
        return False
    return _dragapult_matchup_active(observation) and not _dragapult_evolution_attack_ready(observation, evolution_id, option=option)


def _dragapult_promotion_score(observation, option):
    if not _dragapult_matchup_active(observation):
        return None
    card = _card_from_option(observation, option)
    identifier = _card_id(card)
    if identifier is None:
        return -100_000
    if identifier == ARTICUNO:
        return -1_000_000
    energy_count = _attached_energy_cards(card)
    damaged = _card_is_damaged(card)
    attacker_line = identifier in (MURKROW, HONCHKROW, PORYGON, PORYGON2)
    if attacker_line and energy_count > 0 and damaged:
        return 620_000
    if attacker_line and energy_count > 0:
        return 560_000
    if attacker_line and damaged:
        return 380_000
    baseline = {
        MURKROW: 300_000,
        PORYGON: 260_000,
        HONCHKROW: 235_000,
        PORYGON2: 220_000,
    }.get(identifier)
    if baseline is not None:
        return baseline
    return -50_000


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


def _active_card_for_player(current, player_index):
    player = _player_state(current, player_index)
    return _top_card(_read(player, "active", []))


def _is_hop_phantump_dodge_attack_log(entry, current):
    player_index = _safe_int(_read(entry, "playerIndex"), -1)
    if player_index < 0:
        return False

    log_type = _safe_int(_read(entry, "type"), None)
    attack_id = _safe_int(_read(entry, "attackId", _read(entry, "attack_id")), None)
    identifier = _card_id({"id": _read(entry, "cardId", _read(entry, "card_id"))})
    active = _active_card_for_player(current, player_index)
    active_id = _card_id(active)

    if identifier == HOP_PHANTUMP:
        return log_type == 15 or attack_id is not None
    if attack_id in HOP_PHANTUMP_DODGE_ATTACK_IDS and identifier in (None, HOP_PHANTUMP):
        return True
    if log_type == 15 and identifier is None and active_id == HOP_PHANTUMP:
        return True
    return False


def _hop_phantump_active_serial(current, player_index):
    active = _active_card_for_player(current, player_index)
    if _card_id(active) != HOP_PHANTUMP:
        return None
    return _safe_int(_read(active, "serial"), None)


def _remember_hop_phantump_dodge(observation):
    current = _read(observation, "current", {})
    your_index = _safe_int(_read(current, "yourIndex"), -1)
    turn = _safe_int(_read(current, "turn"), -1)
    logs = _read(observation, "logs", [])
    if your_index < 0 or turn < 0 or not isinstance(logs, list):
        return

    pending_by_player = {}
    for entry in logs:
        if not isinstance(entry, dict):
            continue
        player_index = _safe_int(_read(entry, "playerIndex"), -1)
        if player_index < 0:
            continue
        if _is_hop_phantump_dodge_attack_log(entry, current):
            pending_by_player[player_index] = {
                "serial": _safe_int(_read(entry, "serial"), _hop_phantump_active_serial(current, player_index)),
            }
            continue

        if "head" not in entry:
            continue

        pending = pending_by_player.get(player_index)
        if pending is None and _hop_phantump_active_serial(current, player_index) is not None:
            pending = {"serial": _hop_phantump_active_serial(current, player_index)}
        if pending is None:
            continue

        against_player = 1 - player_index if player_index in (0, 1) else your_index
        expires_turn = turn if your_index == against_player else turn + 1

        if bool(_read(entry, "head")):
            HOP_DODGE_MEMORY[player_index] = {
                "against_player": against_player,
                "turn": turn,
                "expires_turn": expires_turn,
                "serial": pending["serial"],
            }
        else:
            existing = HOP_DODGE_MEMORY.get(player_index)
            if (
                isinstance(existing, dict)
                and existing.get("against_player") == against_player
            ):
                HOP_DODGE_MEMORY.pop(player_index, None)
        pending_by_player.pop(player_index, None)


def _record_observed_attack(player_index, card_id, attack_id, turn, damage=0, count_event=False):
    if player_index < 0 or attack_id < 0:
        return
    player_memory = OBSERVED_ATTACK_MEMORY.setdefault(player_index, {"attacks": {}, "byCard": {}})
    attack_memory = player_memory["attacks"].setdefault(
        attack_id,
        {"count": 0, "maxDamage": 0, "lastTurn": -1},
    )
    if count_event:
        attack_memory["count"] = attack_memory.get("count", 0) + 1
        attack_memory["lastTurn"] = max(_safe_int(attack_memory.get("lastTurn"), -1), turn)
    attack_memory["maxDamage"] = max(_safe_int(attack_memory.get("maxDamage"), 0), damage)

    if card_id is None:
        return
    card_memory = player_memory["byCard"].setdefault(card_id, {})
    card_attack_memory = card_memory.setdefault(
        attack_id,
        {"count": 0, "maxDamage": 0, "lastTurn": -1},
    )
    if count_event:
        card_attack_memory["count"] = card_attack_memory.get("count", 0) + 1
        card_attack_memory["lastTurn"] = max(_safe_int(card_attack_memory.get("lastTurn"), -1), turn)
    card_attack_memory["maxDamage"] = max(_safe_int(card_attack_memory.get("maxDamage"), 0), damage)


def _remember_observed_attacks(observation):
    current = _read(observation, "current", {})
    turn = _safe_int(_read(current, "turn"), -1)
    logs = _read(observation, "logs", [])
    if not isinstance(logs, list):
        return

    last_attack = None
    for entry in logs:
        if not isinstance(entry, dict):
            continue
        log_type = _read(entry, "type")
        if log_type == 15:
            player_index = _safe_int(_read(entry, "playerIndex"), -1)
            attack_id = _safe_int(_read(entry, "attackId"), -1)
            card_id = _safe_int(_read(entry, "cardId"), None)
            serial = _safe_int(_read(entry, "serial"), None)
            if player_index < 0 or attack_id < 0:
                last_attack = None
                continue
            key = (player_index, card_id, serial, attack_id, turn)
            is_new = key not in OBSERVED_ATTACK_LOG_KEYS
            if is_new:
                OBSERVED_ATTACK_LOG_KEYS.add(key)
            _record_observed_attack(player_index, card_id, attack_id, turn, count_event=is_new)
            last_attack = (player_index, card_id, attack_id)
            continue

        if log_type != 16 or last_attack is None:
            continue
        value = _safe_int(_read(entry, "value"), 0)
        if value == 0:
            continue
        damage = abs(value)
        player_index, card_id, attack_id = last_attack
        _record_observed_attack(player_index, card_id, attack_id, turn, damage=damage, count_event=False)


def _opponent_active_has_hop_dodge_protection(observation):
    current = _read(observation, "current", {})
    your_index = _safe_int(_read(current, "yourIndex"), -1)
    turn = _safe_int(_read(current, "turn"), -1)
    if your_index < 0 or turn < 0:
        return False

    opponent_index = 1 - your_index
    opponent = _player_state(current, opponent_index)
    active = _top_card(_read(opponent, "active", []))
    if _card_id(active) != HOP_PHANTUMP:
        return False

    memory = HOP_DODGE_MEMORY.get(opponent_index)
    if not isinstance(memory, dict):
        return False
    if memory.get("against_player") != your_index:
        return False
    expires_turn = _safe_int(memory.get("expires_turn"), _safe_int(memory.get("turn"), turn))
    if turn > expires_turn:
        HOP_DODGE_MEMORY.pop(opponent_index, None)
        return False
    if turn < _safe_int(memory.get("turn"), turn) - 1:
        return False

    memory_serial = memory.get("serial")
    active_serial = _safe_int(_read(active, "serial"), None)
    if memory_serial is not None and active_serial is not None and memory_serial != active_serial:
        return False
    return True


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
    table_prize_count = PRIZE_COUNT_BY_CARD_ID.get(identifier)
    if table_prize_count is not None:
        return table_prize_count

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
    if re.search(r"\bmega\b", text) and re.search(r"\bex\b|ex$", text):
        return 3
    if re.search(r"\bex\b|ex$", text):
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
    needs_evolution: bool = False
    evolution_id: Any = None


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


def _planned_attack_damage_for_id(player, attacker_id, target, attack_id):
    if attacker_id == HONCHKROW and attack_id == ROCKET_FEATHER_ATTACK:
        hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)
        if hand_supporters <= 0:
            return 0
        return _rocket_feather_damage_per_supporter(target) * hand_supporters
    if attacker_id == PORYGON2 and attack_id in PORYGON2_R_COMMAND_ATTACKS:
        discard_supporters = _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS)
        damage = discard_supporters * _policy_rule_number("porygon2RCommandFallback", "damagePerTrashRocketSupporter", 20)
        return _damage_after_type_modifier(damage, target, "colorless")
    if attacker_id == MURKROW and attack_id in MURKROW_ATTACKS:
        return 0
    return 0


def _planned_attack_damage(player, attacker, target, attack_id):
    return _planned_attack_damage_for_id(player, _card_id(attacker), target, attack_id)


def _planned_attack_ids(attacker):
    attacker_id = _card_id(attacker)
    if attacker_id == HONCHKROW:
        return (ROCKET_FEATHER_ATTACK,)
    if attacker_id == PORYGON2:
        return tuple(PORYGON2_R_COMMAND_ATTACKS)
    if attacker_id == MURKROW:
        return ()
    return ()


def _only_ariana_rocket_feather_fuel(player):
    hand_supporter_ids = [
        _card_id(card)
        for card in _iter_cards(_read(player, "hand", []))
        if _card_id(card) in ROCKET_SUPPORTERS
    ]
    return len(hand_supporter_ids) == 1 and hand_supporter_ids[0] == ARIANA


def _non_ko_refill_supporter_to_preserve(player):
    hand_supporter_ids = [
        _card_id(card)
        for card in _iter_cards(_read(player, "hand", []))
        if _card_id(card) in ROCKET_SUPPORTERS
    ]
    if ARIANA in hand_supporter_ids:
        return ARIANA
    if ARCHER in hand_supporter_ids:
        return ARCHER
    return None


def _non_ko_rocket_feather_fuel_count(player, limit=1):
    hand_supporter_ids = [
        _card_id(card)
        for card in _iter_cards(_read(player, "hand", []))
        if _card_id(card) in ROCKET_SUPPORTERS
    ]
    preserved = _non_ko_refill_supporter_to_preserve(player)
    if preserved is None:
        expendable = len(hand_supporter_ids)
    else:
        preserved_count = hand_supporter_ids.count(preserved)
        expendable = sum(1 for card_id in hand_supporter_ids if card_id != preserved)
        expendable += max(0, preserved_count - 1)
    return min(max(0, limit), expendable)


def _non_ko_rocket_feather_split_fuel_count(player, remaining_hp, damage_per_supporter, limit=1):
    baseline = _non_ko_rocket_feather_fuel_count(player, 1)
    preserved = _non_ko_refill_supporter_to_preserve(player)
    if preserved is None:
        return baseline
    expendable = _non_ko_rocket_feather_fuel_count(player, limit)
    if expendable <= baseline or remaining_hp <= 0 or damage_per_supporter <= 0:
        return baseline

    damage = damage_per_supporter * expendable
    remaining_after = max(0, remaining_hp - damage)
    next_rocket_feather_reachable = remaining_after <= damage_per_supporter * 4
    meaningful_split_progress = damage >= max(damage_per_supporter * 2, (remaining_hp + 1) // 2)
    if next_rocket_feather_reachable or meaningful_split_progress:
        return expendable
    return baseline


def _non_ko_cost_candidates_preserving_ariana(candidates):
    protected = ARIANA if any(item[2] == ARIANA for item in candidates) else None
    if protected is None and any(item[2] == ARCHER for item in candidates):
        protected = ARCHER
    if protected is None:
        return list(candidates)
    keep_protected = True
    filtered = []
    for item in candidates:
        if item[2] == protected and keep_protected:
            keep_protected = False
            continue
        filtered.append(item)
    return filtered


def _rocket_feather_cost_candidates_preserving_refill(candidates, required_count):
    protected_candidates = _non_ko_cost_candidates_preserving_ariana(candidates)
    if len(protected_candidates) >= max(0, required_count):
        return protected_candidates
    return list(candidates)


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


def _planned_evolved_attacker_needs_energy(base_card, evolved_id, attack_id):
    energy_cards = _attached_energy_cards(base_card)
    energy_ids = _attached_energy_card_ids(base_card)
    if evolved_id == HONCHKROW and attack_id == ROCKET_FEATHER_ATTACK:
        return energy_cards <= 0
    if evolved_id == PORYGON2 and attack_id in PORYGON2_R_COMMAND_ATTACKS:
        return energy_cards <= 0 and IGNITION_ENERGY not in energy_ids
    return False


def _planned_energy_available(player, attacker, attack_id):
    return _planned_energy_available_for_id(player, _card_id(attacker), attack_id)


def _planned_energy_available_for_id(player, attacker_id, attack_id):
    hand_ids = [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]
    if attacker_id == HONCHKROW and attack_id == ROCKET_FEATHER_ATTACK:
        return TEAM_ROCKET_ENERGY in hand_ids or IGNITION_ENERGY in hand_ids
    if attacker_id == PORYGON2 and attack_id in PORYGON2_R_COMMAND_ATTACKS:
        return IGNITION_ENERGY in hand_ids
    if attacker_id == MURKROW and attack_id in MURKROW_ATTACKS:
        return TEAM_ROCKET_ENERGY in hand_ids
    return False


def _planned_evolution_attack_ids(base_id, evolved_id):
    if base_id == MURKROW and evolved_id == HONCHKROW:
        return (ROCKET_FEATHER_ATTACK,)
    if base_id == PORYGON and evolved_id == PORYGON2:
        return tuple(PORYGON2_R_COMMAND_ATTACKS)
    return ()


def _meaningful_evolution_attack_bridge(can_ko, prize_count, damage, target_hp):
    if can_ko:
        return True
    if prize_count <= 1:
        return False
    return damage > 0 and damage * 2 >= target_hp


def _make_turn_plan(
    player,
    attacker,
    attacker_id,
    attacker_index,
    target,
    attack_id,
    damage,
    target_hp,
    prize_count,
    needs_energy,
    needs_switch,
    seed_guard_blocked,
    needs_evolution=False,
    evolution_id=None,
):
    can_ko = damage >= target_hp
    score, game_end = _turn_plan_score(
        player,
        target,
        damage,
        can_ko,
        prize_count,
        needs_energy,
        needs_switch,
        seed_guard_blocked,
    )
    if needs_evolution:
        score += _policy_rule_number("donkrowTurnPlan", "evolutionBridgeScore", 18_000)
        if can_ko:
            score += _policy_rule_number("donkrowTurnPlan", "evolutionBridgeKoBonus", 22_000)
        elif prize_count >= 2 and damage * 2 >= target_hp:
            score += _policy_rule_number("donkrowTurnPlan", "evolutionBridgeTwoTurnBonus", 8_000)
    return DonkrowTurnPlan(
        attacker_card=attacker,
        attacker_id=attacker_id,
        attacker_index=attacker_index,
        target_card=target,
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
        needs_evolution=needs_evolution,
        evolution_id=evolution_id,
    )

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


def _porygon2_late_trash_threshold():
    return _policy_rule_number("porygon2RCommandFallback", "lateTrashThreshold", 15)


def _porygon2_late_trash_bonus(discard_supporters, damage, can_ko=False):
    threshold = _porygon2_late_trash_threshold()
    if discard_supporters < threshold:
        return 0
    score = _policy_rule_number("porygon2RCommandFallback", "lateTrashAttackBonus", 36_000)
    score += max(0, discard_supporters - threshold) * _policy_rule_number(
        "porygon2RCommandFallback",
        "lateTrashPerExtraSupporterScore",
        1_400,
    )
    score += max(0, min(max(0, damage), 420) - 180) * _policy_rule_number(
        "porygon2RCommandFallback",
        "lateTrashHighDamageScore",
        24,
    )
    if can_ko:
        score += _policy_rule_number("porygon2RCommandFallback", "lateTrashKoBonus", 18_000)
    return score


def _porygon2_endgame_prize_threshold():
    return _policy_rule_number("porygon2RCommandFallback", "endgamePrizeThreshold", 3)


def _porygon2_endgame_r_command_ko_bonus(player, can_ko, over_rocket_feather=False):
    if not can_ko or _prize_cards_remaining(player) > _porygon2_endgame_prize_threshold():
        return 0
    bonus = _policy_rule_number("porygon2RCommandFallback", "endgameKoBonus", 46_000)
    if over_rocket_feather:
        bonus += _policy_rule_number("porygon2RCommandFallback", "endgameKoOverRocketFeatherBonus", 82_000)
    if _deck_count(player) <= _policy_rule_number("porygon2RCommandFallback", "lowDeckThreshold", 10):
        bonus += _policy_rule_number("porygon2RCommandFallback", "endgameLowDeckKoBonus", 24_000)
    return bonus


def _porygon2_endgame_r_command_ko_window(player):
    return _prize_cards_remaining(player) <= _porygon2_endgame_prize_threshold()


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
    options = _select_options(select)

    opponent_active = _opponent_active_card(observation)
    if opponent_active is None:
        return None
    target_hp = _remaining_hp(opponent_active)
    if target_hp <= 0:
        return None

    discard_supporters = _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    endgame_r_command_window = _porygon2_endgame_r_command_ko_window(player)
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

    def consider_plan(plan):
        nonlocal best_plan
        if plan is None:
            return
        if best_plan is None or plan.score > best_plan.score:
            best_plan = plan

    def rocket_feather_ko_plan_exists():
        for attacker_index, attacker, needs_switch in attackers:
            if _card_id(attacker) != HONCHKROW:
                continue
            if needs_switch and not can_switch:
                continue
            attack_id = ROCKET_FEATHER_ATTACK
            needs_energy = _planned_attacker_needs_energy(attacker, attack_id)
            if needs_energy and (energy_attached or not _planned_energy_available(player, attacker, attack_id)):
                continue
            if _planned_attack_damage(player, attacker, opponent_active, attack_id) >= target_hp:
                return True
        return False

    endgame_rocket_feather_ko_exists = endgame_r_command_window and rocket_feather_ko_plan_exists()

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
            plan = _make_turn_plan(
                player,
                attacker,
                _card_id(attacker),
                attacker_index,
                opponent_active,
                attack_id,
                damage,
                target_hp,
                prize_count,
                needs_energy,
                needs_switch,
                seed_guard_blocked,
            )
            attacker_id = _card_id(attacker)
            if attacker_id == HONCHKROW:
                plan = replace(plan, score=plan.score + _policy_rule_number("donkrowTurnPlan", "honchkrowPlanBonus", 12_000))
            elif attacker_id == MURKROW:
                score = plan.score - _policy_rule_number("donkrowTurnPlan", "murkrowAttackPlanPenalty", 28_000)
                if not plan.can_ko:
                    score -= _policy_rule_number("donkrowTurnPlan", "murkrowNonKoPenalty", 18_000)
                plan = replace(plan, score=score)
            elif attacker_id == PORYGON2:
                if discard_supporters >= _porygon2_late_trash_threshold():
                    score = plan.score + _policy_rule_number("porygon2RCommandFallback", "lateTrashPlanBonus", 52_000)
                    score += _porygon2_late_trash_bonus(discard_supporters, damage, plan.can_ko)
                    plan = replace(plan, score=score)
                else:
                    score = plan.score
                    if not plan.can_ko:
                        score -= max(_policy_rule_number("donkrowTurnPlan", "porygon2NonKoPenalty", 8_000), 28_000)
                    if _honchkrow_chain_available(player) and not (plan.can_ko and prize_count >= 2):
                        score -= 18_000
                    plan = replace(plan, score=score)
                if plan.can_ko:
                    score = plan.score + _porygon2_endgame_r_command_ko_bonus(
                        player,
                        plan.can_ko,
                        over_rocket_feather=endgame_rocket_feather_ko_exists,
                    )
                    plan = replace(plan, score=score)
            consider_plan(plan)

    hand_ids = _rule_hand_ids(player)
    for option in options:
        if _option_type(option) not in (9, "evolve"):
            continue
        evolved_id = _rule_option_id(observation, option)
        base_card = _target_card_from_option(observation, option)
        base_id = _card_id(base_card)
        if evolved_id not in hand_ids:
            continue
        if evolved_id == PORYGON2 and not _porygon_development_allowed(player):
            continue
        evolved_index = 0
        if isinstance(_read(player, "bench", []), list):
            for bench_index, slot in enumerate(_read(player, "bench", [])):
                if _same_card_instance(_top_card(slot), base_card):
                    evolved_index = bench_index + 1
                    break
        needs_switch = evolved_index != 0
        if needs_switch and not can_switch:
            continue
        for attack_id in _planned_evolution_attack_ids(base_id, evolved_id):
            needs_energy = _planned_evolved_attacker_needs_energy(base_card, evolved_id, attack_id)
            if needs_energy and (energy_attached or not _planned_energy_available_for_id(player, evolved_id, attack_id)):
                continue
            damage = _planned_attack_damage_for_id(player, evolved_id, opponent_active, attack_id)
            if damage <= 0:
                continue
            prize_count = _prize_count_for_knockout(opponent_active)
            can_ko = damage >= target_hp
            if not _meaningful_evolution_attack_bridge(can_ko, prize_count, damage, target_hp):
                continue
            plan = _make_turn_plan(
                player,
                base_card,
                evolved_id,
                evolved_index,
                opponent_active,
                attack_id,
                damage,
                target_hp,
                prize_count,
                needs_energy,
                needs_switch,
                seed_guard_blocked,
                needs_evolution=True,
                evolution_id=evolved_id,
            )
            if evolved_id == HONCHKROW:
                plan = replace(plan, score=plan.score + _policy_rule_number("donkrowTurnPlan", "honchkrowPlanBonus", 12_000))
            elif evolved_id == PORYGON2:
                score = plan.score
                if discard_supporters >= _porygon2_late_trash_threshold():
                    score += _policy_rule_number("porygon2RCommandFallback", "lateTrashPlanBonus", 52_000)
                    score += _porygon2_late_trash_bonus(discard_supporters, damage, can_ko)
                elif not can_ko:
                    score -= max(_policy_rule_number("donkrowTurnPlan", "porygon2NonKoPenalty", 8_000), 28_000)
                if _honchkrow_chain_available(player) and not (can_ko and prize_count >= 2):
                    score -= 18_000
                if can_ko:
                    score += _porygon2_endgame_r_command_ko_bonus(
                        player,
                        can_ko,
                        over_rocket_feather=endgame_rocket_feather_ko_exists,
                    )
                plan = replace(plan, score=score)
            consider_plan(plan)
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
        return _rocket_feather_damage_per_supporter(target) * fuel_after_sakaki
    if identifier == PORYGON2:
        discard_supporters = _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS)
        damage = (discard_supporters + 1) * _policy_rule_number("porygon2RCommandFallback", "damagePerTrashRocketSupporter", 20)
        return _damage_after_type_modifier(damage, target, "colorless")
    return 0


def _bench_candidate_damage_after_sakaki(observation, attacker, target, giovanni_from_hand=True):
    current, _, player = _current_player(observation)
    identifier = _card_id(attacker)
    hand_ids = [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]
    hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    fuel_after_sakaki = max(0, hand_supporters - (1 if giovanni_from_hand else 0))
    energy_already_used = _energy_attached_this_turn(current, player)
    energy_cards = _attached_energy_cards(attacker)
    energy_ids = _attached_energy_card_ids(attacker)

    if identifier == HONCHKROW:
        if energy_cards <= 0 and (energy_already_used or not (TEAM_ROCKET_ENERGY in hand_ids or IGNITION_ENERGY in hand_ids)):
            return 0
        if fuel_after_sakaki <= 0:
            return 0
        return _rocket_feather_damage_per_supporter(target) * fuel_after_sakaki

    if identifier == MURKROW:
        if HONCHKROW not in hand_ids:
            return 0
        if bool(_read(attacker, "appearThisTurn", False)):
            return 0
        if energy_cards <= 0 and (energy_already_used or not (TEAM_ROCKET_ENERGY in hand_ids or IGNITION_ENERGY in hand_ids)):
            return 0
        if fuel_after_sakaki <= 0:
            return 0
        return _rocket_feather_damage_per_supporter(target) * fuel_after_sakaki

    if identifier == PORYGON2:
        if energy_cards <= 0 and IGNITION_ENERGY not in energy_ids and (energy_already_used or IGNITION_ENERGY not in hand_ids):
            return 0
        discard_supporters = _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS)
        damage = (discard_supporters + 1) * _policy_rule_number("porygon2RCommandFallback", "damagePerTrashRocketSupporter", 20)
        return _damage_after_type_modifier(damage, target, "colorless")

    return 0


def _active_candidate_damage_without_sakaki(observation, target):
    if target is None or _opponent_active_has_hop_dodge_protection(observation):
        return 0

    current, _, player = _current_player(observation)
    active = _top_card(_read(player, "active", []))
    identifier = _card_id(active)
    hand_ids = [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]
    hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    energy_already_used = _energy_attached_this_turn(current, player)
    energy_cards = _attached_energy_cards(active)
    energy_ids = _attached_energy_card_ids(active)

    if identifier == HONCHKROW:
        if energy_cards <= 0 and (energy_already_used or not (TEAM_ROCKET_ENERGY in hand_ids or IGNITION_ENERGY in hand_ids)):
            return 0
        if hand_supporters <= 0:
            return 0
        return _rocket_feather_damage_per_supporter(target) * hand_supporters

    if identifier == PORYGON2:
        if energy_cards <= 0 and IGNITION_ENERGY not in energy_ids and (energy_already_used or IGNITION_ENERGY not in hand_ids):
            return 0
        discard_supporters = _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS)
        damage = discard_supporters * _policy_rule_number("porygon2RCommandFallback", "damagePerTrashRocketSupporter", 20)
        return _damage_after_type_modifier(damage, target, "colorless")

    return 0


def _active_high_prize_pressure_without_sakaki(observation, min_prize=2):
    target = _opponent_active_card(observation)
    if target is None or _opponent_active_has_hop_dodge_protection(observation):
        return False
    if _prize_count_for_knockout(target) < min_prize:
        return False

    remaining_hp = _remaining_hp(target)
    if remaining_hp <= 0:
        return False

    damage = _active_candidate_damage_without_sakaki(observation, target)
    if damage <= 0:
        return False
    if damage >= remaining_hp:
        return True

    progress_threshold = min(180, max(60, (remaining_hp + 1) // 2))
    return damage >= progress_threshold


def _active_prize_count_if_ko_without_sakaki(observation):
    target = _opponent_active_card(observation)
    remaining_hp = _remaining_hp(target)
    if remaining_hp <= 0:
        return 0
    damage = _active_candidate_damage_without_sakaki(observation, target)
    if damage >= remaining_hp:
        return _prize_count_for_knockout(target)
    return 0


def _sakaki_bench_prize_pressure(observation):
    if not _policy_rule_enabled("sakakiRequiresKo"):
        return False
    _, _, player = _current_player(observation)
    hand_ids = [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]
    if GIOVANNI not in hand_ids:
        return False
    return _sakaki_prize_race_ko_score(observation, giovanni_from_hand=True) is not None


def _sakaki_can_take_remaining_prizes(observation, giovanni_from_hand=True):
    if not _policy_rule_enabled("sakakiRequiresKo"):
        return False
    current, _, player = _current_player(observation)
    if giovanni_from_hand and GIOVANNI not in _rule_hand_ids(player):
        return False
    active = _top_card(_read(player, "active", []))
    if _card_id(active) not in ROCKET_FIELD_POKEMON:
        return False

    remaining_prizes = max(1, _prize_cards_remaining(player))
    active_prizes_without_sakaki = _active_prize_count_if_ko_without_sakaki(observation)
    if active_prizes_without_sakaki >= remaining_prizes:
        return False

    bench_attackers = [
        pokemon
        for slot in _read(player, "bench", [])
        if (pokemon := _top_card(slot)) is not None and _card_id(pokemon) in (HONCHKROW, MURKROW, PORYGON2)
    ]
    if not bench_attackers:
        return False

    for target in _opponent_bench_cards(observation):
        remaining_hp = _remaining_hp(target)
        prize_count = _prize_count_for_knockout(target)
        if remaining_hp <= 0 or prize_count < remaining_prizes:
            continue
        for attacker in bench_attackers:
            damage = _bench_candidate_damage_after_sakaki(observation, attacker, target, giovanni_from_hand)
            if damage >= remaining_hp:
                return True
    return False


def _sakaki_prize_race_ko_score(observation, giovanni_from_hand=True):
    if not _policy_rule_enabled("sakakiRequiresKo"):
        return None
    current, your_index, player = _current_player(observation)
    active = _top_card(_read(player, "active", []))
    if _card_id(active) not in ROCKET_FIELD_POKEMON:
        return None

    min_prize = _policy_rule_number("sakakiRequiresKo", "minPrizeScore", 2)
    active_prizes_without_sakaki = _active_prize_count_if_ko_without_sakaki(observation)
    if active_prizes_without_sakaki >= min_prize:
        return None
    active_high_prize_pressure = _active_high_prize_pressure_without_sakaki(observation, min_prize=min_prize)

    bench_attackers = [
        pokemon
        for slot in _read(player, "bench", [])
        if (pokemon := _top_card(slot)) is not None and _card_id(pokemon) in (HONCHKROW, MURKROW, PORYGON2)
    ]
    if not bench_attackers:
        return None

    targets = []
    for target in _opponent_bench_cards(observation):
        remaining_hp = _remaining_hp(target)
        prize_count = _prize_count_for_knockout(target)
        if remaining_hp <= 0 or prize_count <= 0:
            continue
        if prize_count < min_prize and (active_prizes_without_sakaki > 0 or active_high_prize_pressure):
            continue
        if remaining_hp > 0:
            targets.append((target, remaining_hp, prize_count))
    if not targets:
        return None

    best_score = None
    for attacker in bench_attackers:
        for target, remaining_hp, prize_count in targets:
            damage = _bench_candidate_damage_after_sakaki(observation, attacker, target, giovanni_from_hand)
            if damage < remaining_hp:
                continue
            score = (
                _policy_rule_number("sakakiRequiresKo", "koBaseScore", 95_000) +
                prize_count * _policy_rule_number("sakakiRequiresKo", "koPerPrizeScore", 75_000) +
                min(damage, 360) * 12
            )
            if prize_count < min_prize:
                score -= 35_000
            if _card_id(attacker) == HONCHKROW:
                score += 4_500
            if _card_id(attacker) == MURKROW:
                score += 2_500
            if prize_count >= 3:
                score += 20_000
            if best_score is None or score > best_score:
                best_score = score
    return best_score


def _sakaki_hop_dodge_escape_ko_score(observation, giovanni_from_hand=True):
    if not _opponent_active_has_hop_dodge_protection(observation):
        return None
    return _sakaki_prize_race_ko_score(observation, giovanni_from_hand=giovanni_from_hand)


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
        or (MURKROW in field_ids and (HONCHKROW in hand_ids or _deck_card_count_for_policy(player, lambda card_id: card_id == HONCHKROW) > 0))
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
        energy_preference = _supporter_energy_dig_preference(current, player, apollo_from_hand=apollo_from_hand)
        if needs_energy and energy_preference is not None and energy_preference[0] == "apollo":
            return False
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
    deck_energy = _deck_card_count_for_policy(player, lambda card_id: card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY))
    deck_evolutions = _deck_card_count_for_policy(player, lambda card_id: card_id in (HONCHKROW, PORYGON2))
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
    deck_energy = _deck_card_count_for_policy(player, lambda card_id: card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY))
    if hand_energy > 0 or deck_energy <= 0:
        return False
    active = _top_card(_read(player, "active", []))
    active_id = _card_id(active)
    active_needs_energy = active_id in (HONCHKROW, PORYGON2, MURKROW) and _attached_energy_cards(active) <= 0
    field_ids = _field_card_ids(player)
    has_attack_line = any(card_id in (HONCHKROW, PORYGON2, MURKROW, PORYGON) for card_id in field_ids)
    return active_needs_energy or has_attack_line


def _energy_attached_this_turn(current, player):
    return bool(
        _read(player, "energyAttached", False)
        or _read(player, "energyAttachedThisTurn", False)
        or _read(player, "energy_attached", False)
        or _read(current, "energyAttached", False)
        or _read(current, "energyAttachedThisTurn", False)
        or _read(current, "energy_attached", False)
    )


def _energy_hit_probability(population_count, energy_count, draw_count):
    population_count = max(0, _safe_int(population_count, 0))
    energy_count = max(0, min(population_count, _safe_int(energy_count, 0)))
    draw_count = max(0, min(population_count, _safe_int(draw_count, 0)))
    if population_count <= 0 or energy_count <= 0 or draw_count <= 0:
        return 0.0

    miss_probability = 1.0
    non_energy_count = population_count - energy_count
    for draw_index in range(draw_count):
        remaining = population_count - draw_index
        remaining_non_energy = non_energy_count - draw_index
        if remaining_non_energy <= 0:
            return 1.0
        miss_probability *= remaining_non_energy / remaining
    return 1.0 - miss_probability


def _hit_at_least_probability(population_count, hit_count, draw_count, need_count):
    population_count = max(0, _safe_int(population_count, 0))
    hit_count = max(0, min(population_count, _safe_int(hit_count, 0)))
    draw_count = max(0, min(population_count, _safe_int(draw_count, 0)))
    need_count = max(0, _safe_int(need_count, 0))
    if need_count <= 0:
        return 1.0
    if population_count <= 0 or hit_count <= 0 or draw_count <= 0 or hit_count < need_count:
        return 0.0
    if draw_count >= population_count:
        return 1.0 if hit_count >= need_count else 0.0

    total = math.comb(population_count, draw_count)
    if total <= 0:
        return 0.0
    miss_or_short = 0
    max_short_hits = min(need_count - 1, hit_count, draw_count)
    non_hit_count = population_count - hit_count
    for drawn_hits in range(max_short_hits + 1):
        drawn_non_hits = draw_count - drawn_hits
        if drawn_non_hits < 0 or drawn_non_hits > non_hit_count:
            continue
        miss_or_short += math.comb(hit_count, drawn_hits) * math.comb(non_hit_count, drawn_non_hits)
    return max(0.0, min(1.0, 1.0 - miss_or_short / total))


def _supporter_energy_dig_odds(player, apollo_from_hand=True):
    hand_energy = _count_cards(player, ("hand",), lambda card_id: card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY))
    deck_energy = _deck_card_count_for_policy(player, lambda card_id: card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY))
    deck_count = _deck_count(player)
    if hand_energy > 0 or deck_energy <= 0 or deck_count <= 0:
        return None

    hand_count = _hand_count(player)
    ariana_draw_count = _ariana_draw_count_for_player(player)
    ariana_probability = _energy_hit_probability(deck_count, deck_energy, ariana_draw_count)

    apollo_shuffle_count = max(0, hand_count - (1 if apollo_from_hand else 0))
    apollo_population = deck_count + apollo_shuffle_count
    apollo_draw_count = min(5, apollo_population)
    apollo_probability = _energy_hit_probability(apollo_population, deck_energy, apollo_draw_count)

    return {
        "ariana_probability": ariana_probability,
        "apollo_probability": apollo_probability,
        "ariana_draw_count": ariana_draw_count,
        "apollo_draw_count": apollo_draw_count,
    }


def _supporter_energy_dig_preference(current, player, apollo_from_hand=True):
    if _supporter_played_this_turn(current, player) or _energy_attached_this_turn(current, player):
        return None
    if not _needs_ariana_energy_dig(player):
        return None

    odds = _supporter_energy_dig_odds(player, apollo_from_hand=apollo_from_hand)
    if odds is None:
        return None

    margin = 0.001
    if odds["apollo_probability"] > odds["ariana_probability"] + margin:
        return "apollo", odds
    return "ariana", odds


def _supporter_energy_dig_score_adjustment(current, player, identifier, apollo_from_hand=True):
    if identifier not in (ARIANA, ARCHER):
        return 0

    preference = _supporter_energy_dig_preference(current, player, apollo_from_hand=apollo_from_hand)
    if preference is None:
        return 0

    preferred, odds = preference
    probability_gap = abs(odds["apollo_probability"] - odds["ariana_probability"])
    draw_gap = max(0, odds["apollo_draw_count"] - odds["ariana_draw_count"])
    swing = min(34_000, 10_000 + int(probability_gap * 42_000) + draw_gap * 1_200)

    if preferred == "apollo":
        return swing if identifier == ARCHER else -swing
    return min(10_000, swing) if identifier == ARIANA else -min(12_000, swing)


def _rule_board_collapse_reset_needed(observation):
    current, _, player = _current_player(observation)
    hand_count = _hand_count(player)
    deck_count = _deck_count(player)
    remaining_deck_after_reset = deck_count + max(0, hand_count - 1)
    if remaining_deck_after_reset <= _policy_rule_number("rocketApolloReset", "lowDeckResetThreshold", 6):
        return False

    field_cards = _field_top_cards(player)
    field_count = len(field_cards)
    bench_count = _bench_top_count(player)
    if field_count <= 0:
        return False

    active = _top_card(_read(player, "active", []))
    active_id = _card_id(active)
    hand_ids = _rule_hand_ids(player)
    hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)

    # Do not reset away a live attack.  This keeps Apollo from stealing tempo
    # when the turn can already convert into damage or prizes.
    if _rule_find_immediate_ko_attack_option(observation, _select_options(_read(observation, "select", {}))) is not None:
        return False
    if _has_any_main_attack_option(observation) and active_id in (HONCHKROW, PORYGON2) and hand_supporters > 0:
        return False

    has_basic_in_hand = any(card_id in (MURKROW, PORYGON) for card_id in hand_ids)
    active_murkrow_can_evolve = (
        active_id == MURKROW
        and HONCHKROW in hand_ids
        and not bool(_read(active, "appearThisTurn", False))
    )
    hand_energy = any(card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY) for card_id in hand_ids)
    useful_evolution_in_hand = active_murkrow_can_evolve or (
        active_id == PORYGON
        and PORYGON2 in hand_ids
        and IGNITION_ENERGY in hand_ids
    )

    if field_count <= 1 and not (has_basic_in_hand and (useful_evolution_in_hand or hand_energy)):
        return True
    if bench_count <= 0 and active_id in (MURKROW, PORYGON) and not useful_evolution_in_hand:
        return True
    if hand_count >= 7 and _ariana_draw_count_for_player(player) <= 1 and not (has_basic_in_hand or useful_evolution_in_hand):
        return True
    return False


def _rule_apollo_direct_play_score(observation):
    current, _, player = _current_player(observation)
    if _supporter_played_this_turn(current, player):
        return None

    hand_count = _hand_count(player)
    deck_count = _deck_count(player)
    remaining_deck_after_reset = deck_count + max(0, hand_count - 1)
    if remaining_deck_after_reset <= _policy_rule_number("rocketApolloReset", "lowDeckResetThreshold", 6):
        return None

    hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    hand_supporters_after = max(0, hand_supporters - 1)
    hand_energy = _count_cards(player, ("hand",), lambda card_id: card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY))
    deck_energy = _deck_card_count_for_policy(player, lambda card_id: card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY))
    deck_supporters = _deck_card_count_for_policy(player, lambda card_id: card_id in ROCKET_SUPPORTERS)
    deck_evolutions = _deck_card_count_for_policy(player, lambda card_id: card_id in (HONCHKROW, PORYGON2))
    athena_draw_count = _ariana_draw_count_for_player(player)
    needs_energy = _rocket_attack_needs_energy(player) and hand_energy <= 0 and deck_energy > 0
    needs_fuel = _rocket_feather_fuel_needed(player)

    score = 0
    if hand_count <= 5:
        score += 14_000
    if hand_count >= 7 and athena_draw_count <= 1:
        score += 18_000
    if hand_supporters <= 1:
        score += 9_000
    if hand_supporters_after <= 1 and (deck_supporters > 0 or needs_fuel):
        score += 6_500
    if deck_evolutions > 0:
        score += min(3, deck_evolutions) * 2_000
    if needs_fuel:
        score += 9_500
    refuel_ko_probability = _rule_draw_supporter_refuel_current_ko_probability(observation, ARCHER)
    if refuel_ko_probability >= 0.78:
        score += 30_000 + int(refuel_ko_probability * 18_000)
    if _rule_board_collapse_reset_needed(observation):
        score += 38_000
        if hand_count >= 7:
            score += 8_000
        if hand_supporters_after <= 1:
            score += 5_000

    odds = _supporter_energy_dig_odds(player, apollo_from_hand=True)
    if odds is not None:
        probability_gap = odds["apollo_probability"] - odds["ariana_probability"]
        draw_gap = max(0, odds["apollo_draw_count"] - odds["ariana_draw_count"])
        if needs_energy and probability_gap > 0.001:
            score += 30_000 + int(probability_gap * 42_000) + draw_gap * 1_200
        elif needs_energy and athena_draw_count <= 1 and odds["apollo_probability"] >= 0.20:
            score += 16_000 + draw_gap * 1_000

    if ARIANA in _rule_hand_ids(player) and athena_draw_count >= 3 and not needs_energy and not needs_fuel:
        score -= 18_000
    if hand_supporters_after >= 3 and not needs_energy and not needs_fuel:
        score -= 8_000

    return score


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


def _factory_before_ariana_score(observation):
    if not _policy_rule_enabled("compressBeforeAriana"):
        return None

    current, _, player = _current_player(observation)
    if _supporter_played_this_turn(current, player) or _stadium_played_this_turn(current, player):
        return None

    stadium_ids = [_card_id(card) for card in _iter_cards(_read(current, "stadium", []))]
    if FACTORY in stadium_ids:
        return None

    select = _read(observation, "select", {})
    options = _select_options(select)
    has_factory_play = any(
        _card_id(_card_from_option(observation, option)) == FACTORY and _option_type(option) in (7, "play")
        for option in options
    )
    has_ariana_play = any(
        _card_id(_card_from_option(observation, option)) == ARIANA and _option_type(option) in (7, "play")
        for option in options
    )
    if not has_factory_play or not has_ariana_play:
        return None

    hand_ids = [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]
    if FACTORY not in hand_ids or ARIANA not in hand_ids:
        return None

    hand_count = _hand_count(player)
    target_hand = 8 if _all_field_pokemon_are_rocket(player) else 5
    current_draw = max(0, target_hand - max(0, hand_count - 1))
    factory_first_draw = max(0, target_hand - max(0, hand_count - 2))
    deck_count = _deck_count(player)
    draw_gain = min(deck_count, factory_first_draw) - min(deck_count, current_draw)
    if draw_gain <= 0:
        return None

    score = _policy_rule_number("preferArianaEnergyDig", "factoryBeforeArianaBonus", 34_000)
    score += draw_gain * _policy_rule_number("preferArianaEnergyDig", "factoryBeforeArianaDrawGainBonus", 8_000)
    if current_draw <= _policy_rule_number("compressBeforeAriana", "lowDrawMax", 2) and hand_count >= _policy_rule_number("compressBeforeAriana", "highHandMin", 7):
        score += 4_000
    if _needs_ariana_energy_dig(player):
        score += 4_000
    return score


def _factory_before_ariana_option_index(observation, options):
    if _factory_before_ariana_score(observation) is None:
        return None
    for option_index, option in enumerate(options):
        if _card_id(_card_from_option(observation, option)) == FACTORY and _option_type(option) in (7, "play"):
            return option_index
    return None


def _honchkrow_chain_available(player, hand_ids=None, deck_ids=None):
    if hand_ids is None:
        hand_ids = [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]
    if deck_ids is None:
        deck_ids = _deck_card_ids_for_policy(player)
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


def _honchkrow_pad_bridge_available(player, hand_ids=None, deck_ids=None):
    if hand_ids is None:
        hand_ids = [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]
    if deck_ids is None:
        deck_ids = _deck_card_ids_for_policy(player)
    return TEAM_ROCKET_TRANSCEIVER in hand_ids and PROTON in deck_ids and MURKROW in deck_ids


def _bench_honchkrow_promotion_available(player, hand_ids=None, deck_ids=None):
    if hand_ids is None:
        hand_ids = [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]
    if deck_ids is None:
        deck_ids = _deck_card_ids_for_policy(player)
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


def _proton_not_in_discard_or_trash(player):
    return _count_cards(player, ("discard",), lambda card_id: card_id == PROTON) <= 0


def _proton_opening_allowed(current, player, setup_incomplete):
    return (
        bool(setup_incomplete)
        and not _supporter_played_this_turn(current, player)
        and _rocket_supporter_history_count(player) <= 0
    )


def _turn_number(current):
    return _safe_int(_read(current, "turn", _read(current, "turnNumber", _read(current, "turn_number", 0))), 0)


def _first_player_index(current):
    return _safe_int(_read(current, "firstPlayer", _read(current, "first_player", -1)), -1)


def _opening_turn_order(current, player_index):
    turn = _turn_number(current)
    first_player = _first_player_index(current)
    if turn <= 0 or first_player < 0:
        return None
    if first_player == player_index:
        return "first" if turn <= 1 else None
    return "second" if turn <= 2 else None


def _bench_has_murkrow(player):
    for bench_card in _read(player, "bench", []) or []:
        if _card_id(_top_card(bench_card)) == MURKROW:
            return True
    return False


def _opening_turn_team_rocket_energy_score(current, player, player_index, hand_ids, target_id, target_is_active, target_is_bench, target_energy_cards, target_has_rocket_energy):
    order = _opening_turn_order(current, player_index)
    if order is None:
        return None

    rocket_energy_in_hand = sum(1 for card_id in hand_ids if card_id == TEAM_ROCKET_ENERGY)
    if rocket_energy_in_hand <= 0:
        return None

    active_id = _card_id(_top_card(_read(player, "active", [])))
    has_bench_murkrow = _bench_has_murkrow(player)
    target_is_clean_murkrow = target_id == MURKROW and target_energy_cards <= 0 and not target_has_rocket_energy
    must_attach = 135_000
    forbidden = -180_000

    if order == "first":
        if active_id == MURKROW:
            return must_attach if target_is_active and target_is_clean_murkrow else forbidden
        if has_bench_murkrow:
            return must_attach if target_is_bench and target_is_clean_murkrow else forbidden
        return forbidden

    if rocket_energy_in_hand >= 2:
        return must_attach if target_is_bench and target_is_clean_murkrow else forbidden
    return forbidden


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


def _choose_mulligan_bonus_draw(observation, options, max_count):
    select = _read(observation, "select", {})
    if _select_context(select) != 38 or max_count != 1:
        return None

    numbered_options = []
    for option_index, option in enumerate(options):
        number = _read(option, "number")
        if number is None:
            continue
        numbered_options.append((_safe_int(number, -1), option_index))

    if not numbered_options:
        return None

    numbered_options.sort(key=lambda item: (-item[0], item[1]))
    return [numbered_options[0][1]]


def _choose_rocket_feather_costs(observation, options, max_count):
    select = _read(observation, "select", {})
    effect = _read(select, "effect")
    context = _select_context(select)
    if _card_id(effect) != HONCHKROW and context != 8:
        return None

    current = _read(observation, "current", {})
    your_index = _safe_int(_read(current, "yourIndex"), 0)
    player = _player_state(current, your_index)
    opponent = _player_state(current, 1 - your_index)
    opponent_active = _top_card(_read(opponent, "active", []))
    min_count, _ = _selection_bounds(select)
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

    if min_count == 0 and not can_ko:
        candidates = _non_ko_cost_candidates_preserving_ariana(candidates)
        required = min(
            max_count,
            max(
                min_count,
                _non_ko_rocket_feather_split_fuel_count(
                    player,
                    parsed_remaining_hp,
                    damage_per_supporter,
                    max_count,
                ),
            ),
        )
        if required <= 0:
            return []

    protected = {ARIANA, PETREL}
    if _sakaki_bench_prize_pressure(observation):
        protected.add(GIOVANNI)
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


def _promotion_active_card_score(observation, option):
    current, _, player = _current_player(observation)
    card = _card_from_option(observation, option)
    identifier = _card_id(card)
    if identifier is None:
        return -100_000
    dragapult_score = _dragapult_promotion_score(observation, option)
    if dragapult_score is not None:
        return dragapult_score

    hand_ids = [_card_id(hand_card) for hand_card in _iter_cards(_read(player, "hand", []))]
    deck_ids = _deck_card_ids_for_policy(player)
    hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    discard_supporters = _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    hand_energy = _count_cards(player, ("hand",), lambda card_id: card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY))
    deck_energy = _deck_card_count_for_policy(player, lambda card_id: card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY))
    has_honchkrow_in_hand = HONCHKROW in hand_ids
    has_honchkrow_access = has_honchkrow_in_hand or HONCHKROW in deck_ids
    has_energy_access = hand_energy > 0 or deck_energy > 0
    has_fuel = hand_supporters > 0
    field_top_ids = [_card_id(field_card) for field_card in _field_top_cards(player)]
    has_honchkrow_line = any(card_id in (MURKROW, HONCHKROW) for card_id in field_top_ids)
    porygon2_access = PORYGON2 in hand_ids or PORYGON2 in deck_ids
    ignition_access = IGNITION_ENERGY in hand_ids or IGNITION_ENERGY in deck_ids

    if identifier == HONCHKROW:
        if _attached_energy_cards(card) > 0 and has_fuel:
            return 240_000
        if hand_energy > 0 and has_fuel:
            return 222_000
        if has_energy_access and has_fuel:
            return 196_000
        return 148_000

    if identifier == MURKROW:
        if has_honchkrow_in_hand and hand_energy > 0 and has_fuel:
            return 260_000
        if has_honchkrow_in_hand and has_energy_access and has_fuel:
            return 232_000
        if has_honchkrow_in_hand:
            return 205_000
        if has_honchkrow_access and has_energy_access:
            return 168_000
        if has_honchkrow_access:
            return 132_000
        return 72_000

    if identifier == PORYGON2:
        if not _porygon_development_allowed(player):
            return 22_000
        if discard_supporters >= _porygon2_late_trash_threshold():
            can_attack_or_convert = (
                _attached_energy_cards(card) > 0
                or IGNITION_ENERGY in _attached_energy_card_ids(card)
                or IGNITION_ENERGY in hand_ids
            )
            return 280_000 if can_attack_or_convert else 132_000
        if ignition_access and not has_honchkrow_line:
            return 118_000
        if ignition_access:
            return 84_000
        return 52_000

    if identifier == PORYGON:
        if has_honchkrow_line or has_honchkrow_access:
            return -90_000
        if porygon2_access and ignition_access:
            return 66_000
        if porygon2_access:
            return 42_000
        return 18_000

    return 1_000


def _choose_promotion_active_card(observation, options, max_count):
    if max_count != 1:
        return None
    select = _read(observation, "select", {})
    if _select_context(select) not in (4, "promote active pokemon", "context4"):
        return None
    candidates = []
    for option_index, option in enumerate(options):
        if _option_type(option) not in (3, "card"):
            continue
        candidates.append((_promotion_active_card_score(observation, option), option_index))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_index = candidates[0]
    if best_score <= 0:
        return None
    return [best_index]


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


def _taunt_lock_observed_score(observation, attack_id):
    if attack_id < 0:
        return 0
    current = _read(observation, "current", {})
    your_index = _safe_int(_read(current, "yourIndex"), -1)
    opponent_index = 1 - your_index if your_index in (0, 1) else -1
    opponent_active_id = _card_id(_opponent_active_card(observation))

    def score_stats(stats, base):
        if not isinstance(stats, dict):
            return 0
        return (
            base
            + _safe_int(stats.get("maxDamage"), 0) * 100
            + _safe_int(stats.get("count"), 0) * 1_000
            + _safe_int(stats.get("lastTurn"), -1)
        )

    best = 0
    if opponent_index in OBSERVED_ATTACK_MEMORY:
        player_memory = OBSERVED_ATTACK_MEMORY.get(opponent_index, {})
        by_card = player_memory.get("byCard", {})
        if opponent_active_id is not None:
            best = max(best, score_stats(by_card.get(opponent_active_id, {}).get(attack_id), 2_000_000))
        best = max(best, score_stats(player_memory.get("attacks", {}).get(attack_id), 1_000_000))

    for player_memory in OBSERVED_ATTACK_MEMORY.values():
        best = max(best, score_stats(player_memory.get("attacks", {}).get(attack_id), 100_000))
    return best


def _choose_taunt_move_lock_option(observation, options, max_count):
    if max_count != 1 or not options:
        return None
    if any(_card_from_option(observation, option) is not None for option in options):
        return None

    attack_options = [
        (option_index, _attack_id(option))
        for option_index, option in enumerate(options)
        if _option_type(option) in (13, "attack") and _attack_id(option) >= 0
    ]
    named_options = []
    for option_index, option in enumerate(options):
        name = _move_lock_option_name(option)
        if name:
            named_options.append((option_index, name, _move_lock_option_damage(option)))

    select = _select_payload(observation)
    context = _select_context(select)
    effect_id = _effect_id(observation)
    select_text = str(select).lower()
    is_taunt_lock_context = any(
        marker in select_text
        for marker in ("taunt", "いちゃもん", "move_lock", "move lock", "lock")
    ) or (context == 36 and effect_id == MURKROW and bool(attack_options))
    if not is_taunt_lock_context:
        return None

    if len(attack_options) == 1:
        return [attack_options[0][0]]

    observed_options = [
        (_taunt_lock_observed_score(observation, attack_id), option_index, attack_id)
        for option_index, attack_id in attack_options
    ]
    observed_options = [entry for entry in observed_options if entry[0] > 0]
    if observed_options:
        best = max(observed_options, key=lambda item: (item[0], item[2]))
        return [best[1]]

    damaging_named_options = [entry for entry in named_options if entry[2] > 0]
    if damaging_named_options:
        best = max(damaging_named_options, key=lambda item: (item[2], item[1]))
        return [best[0]]

    if len(attack_options) >= 2:
        return [attack_options[1][0]]
    if named_options:
        best = max(named_options, key=lambda item: (item[2], item[1]))
        return [best[0]]
    return None


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
        alakazam_score = _alakazam_basic_target_score(observation, identifier)
        if alakazam_score is not None:
            return alakazam_score // 400
        if _dragapult_matchup_active(observation):
            dragapult_score = _dragapult_basic_target_score(observation, identifier)
            if dragapult_score is not None:
                return dragapult_score // 400
        if identifier == MURKROW:
            return 1_000 - counts.get(MURKROW, 0) * 40
        if identifier == PORYGON:
            return 760 - counts.get(PORYGON, 0) * 80
        return -1_000

    alakazam_score = _alakazam_basic_target_score(observation, identifier)
    if alakazam_score is not None:
        return alakazam_score // 400

    dragapult_score = _dragapult_basic_target_score(observation, identifier)
    if dragapult_score is not None:
        return dragapult_score // 400

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
    deck_ids = _deck_card_ids_for_policy(player)
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
    supporter_played = _supporter_played_this_turn(current, player)
    first_proton_pending = _proton_not_in_discard_or_trash(player)
    first_proton_search_pending = first_proton_pending and PROTON not in hand_ids and not supporter_played
    early_turn = turn_number <= _policy_rule_number("preferProtonWhenSetupIncomplete", "earlyTurnMax", 4)
    stadium_ids = [_card_id(card) for card in _iter_cards(_read(current, "stadium", []))]
    search_option_ids = [_card_id(_card_from_option(observation, option)) for option in options]
    search_can_take_proton = PROTON in search_option_ids
    energy_dig_needed = _needs_ariana_energy_dig(player)
    petrel_energy_bridge = energy_dig_needed and TEAM_ROCKET_ENERGY in deck_ids and TEAM_ROCKET_ENERGY not in hand_ids
    needs_honchkrow_from_pad, needs_porygon2_from_pad, active_pad_evolution = _poke_pad_evolution_attack_need(player, hand_ids)
    porygon_development_allowed = _porygon_development_allowed(player)
    hand_has_compression = any(
        card_id in (POKEGEAR, POKE_PAD, ROTO_STICK, TEAM_ROCKET_TRANSCEIVER, MURKROW, PORYGON, HONCHKROW, PORYGON2)
        for card_id in hand_ids
    )
    energy_murkrow_needs_honchkrow = _energy_prepared_murkrow_needs_honchkrow(player, hand_ids, deck_ids)
    alakazam_petrel_bridge = _alakazam_petrel_poke_pad_bridge_needed(observation)
    petrel_attack_bridge = (
        energy_murkrow_needs_honchkrow
        or needs_honchkrow_from_pad
        or needs_porygon2_from_pad
        or active_pad_evolution
        or alakazam_petrel_bridge
    )
    active_attack_ready = active_honchkrow and has_rocket_feather_action and hand_supporters > 0
    future_honchkrow_line = (
        active_honchkrow
        or HONCHKROW in field_top_ids
        or (MURKROW in field_top_ids and (HONCHKROW in hand_ids or HONCHKROW in deck_ids))
    )
    needs_rocket_feather_fuel = future_honchkrow_line and hand_supporters < 3
    sakaki_pressure = _sakaki_bench_prize_pressure(observation)
    dragapult_articuno_needed = _dragapult_articuno_search_needed(observation)
    alakazam_articuno_recovery = _alakazam_articuno_recovery_needed(observation)
    alakazam_articuno_petrel_recovery = _alakazam_articuno_petrel_recovery_needed(observation)

    def giovanni_fuel_search_score():
        sakaki_ko_score = _sakaki_prize_race_ko_score(observation, giovanni_from_hand=False)
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
                alakazam_score = _alakazam_basic_target_score(observation, identifier)
                if alakazam_score is not None:
                    score = alakazam_score
                else:
                    dragapult_score = _dragapult_basic_target_score(observation, identifier)
                    if dragapult_score is not None:
                        score = dragapult_score
                    else:
                        murkrow_lines = counts.get(MURKROW, 0) + counts.get(HONCHKROW, 0)
                        porygon_lines = counts.get(PORYGON, 0) + counts.get(PORYGON2, 0)
                        if identifier == MURKROW:
                            score = 2_000 if murkrow_lines < 3 else 250
                        elif identifier == PORYGON:
                            score = 1_700 if porygon_lines < 1 else 220
                        else:
                            score = -10_000
            elif first_proton_search_pending and identifier == PROTON and effect in (POKEGEAR, TEAM_ROCKET_TRANSCEIVER, ROTO_STICK):
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
                alakazam_score = _alakazam_basic_target_score(observation, identifier)
                if alakazam_score is not None:
                    score = alakazam_score
                else:
                    dragapult_score = _dragapult_basic_target_score(observation, identifier)
                    if dragapult_score is not None:
                        score = dragapult_score
                    elif seed_guard_needs_basic and seed_guard_has_basic_option and not _is_basic_setup_pokemon(identifier):
                        score = -100_000
                    elif identifier == MURKROW:
                        score = max(_poke_pad_target_score(observation, option_card), 2_100 if field_murkrow_lines < desired_murkrow_lines else 260)
                        if active_attack_ready and field_murkrow_lines < 3:
                            score = max(score, 40_000)
                        if sakaki_pressure:
                            score = max(score, 52_000 if field_murkrow_lines < 2 else 18_000)
                    elif identifier == HONCHKROW:
                        score = max(
                            _poke_pad_target_score(observation, option_card),
                            4_900 if energy_murkrow_needs_honchkrow else (2_300 if field_murkrow_lines > 0 and not has_honchkrow_in_hand else 420),
                        )
                        if needs_honchkrow_from_pad:
                            score += _policy_rule_number("pokePadEvolutionAttack", "honchkrowWithEnergyInHandScore", 12_000)
                        if active_pad_evolution and identifier == HONCHKROW:
                            score += _policy_rule_number("pokePadEvolutionAttack", "activeEvolutionBonus", 2_800)
                        if sakaki_pressure and MURKROW in field_top_ids:
                            score = max(score, 42_000)
                    elif identifier == PORYGON:
                        score = max(_poke_pad_target_score(observation, option_card), 1_650 if field_porygon_lines < 1 else 220)
                    elif identifier == PORYGON2:
                        if not porygon_development_allowed:
                            score = 260
                        elif discard_supporters >= _porygon2_late_trash_threshold() and field_porygon_lines > 0:
                            score = max(_poke_pad_target_score(observation, option_card), 96_000)
                            if needs_porygon2_from_pad:
                                score += _policy_rule_number("pokePadEvolutionAttack", "porygon2WithIgnitionInHandScore", 11_800)
                            if active_pad_evolution:
                                score += _policy_rule_number("pokePadEvolutionAttack", "activeEvolutionBonus", 2_800)
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
                    if dragapult_articuno_needed:
                        score = 260_000
                    elif PROTON in hand_ids:
                        score -= _policy_rule_number("preferProtonWhenSetupIncomplete", "settledSearchPenalty", 50_000)
                    elif proton_opening_allowed:
                        score += 12_000
                    else:
                        score -= _policy_rule_number("preferProtonWhenSetupIncomplete", "settledSearchPenalty", 50_000)
                    if energy_murkrow_needs_honchkrow:
                        score -= 2_700
                elif identifier == PETREL:
                    if alakazam_articuno_petrel_recovery:
                        score += 260_000
                    elif petrel_energy_bridge:
                        score += 28_000
                    elif petrel_attack_bridge:
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
                    score = 260_000 if alakazam_articuno_petrel_recovery else (54_000 if petrel_energy_bridge else 24_000)
                elif identifier == GIOVANNI:
                    score = max(14_000, giovanni_fuel_search_score())
                elif identifier == PROTON:
                    score = 220_000 if dragapult_articuno_needed else (2_000 if proton_opening_allowed else _policy_rule_number("preferProtonWhenSetupIncomplete", "settledRecoveryScore", -50_000))
            elif effect == TEAM_ROCKET_TRANSCEIVER:
                has_proton_in_hand = PROTON in hand_ids
                has_ariana_in_hand = ARIANA in hand_ids
                if identifier == PROTON and dragapult_articuno_needed:
                    score = 260_000
                elif has_ariana_in_hand:
                    if identifier == ARIANA:
                        score = -8_000
                    elif identifier == PETREL:
                        score = 260_000 if alakazam_articuno_petrel_recovery else (72_000 if petrel_energy_bridge else 18_000 + (5_200 if petrel_attack_bridge else 0))
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
                        score = 260_000 if alakazam_articuno_petrel_recovery else (68_000 if petrel_energy_bridge else 12_000 + (5_200 if petrel_attack_bridge else 0))
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
                    if alakazam_articuno_petrel_recovery:
                        score += 260_000
                    elif petrel_energy_bridge and not supporter_played:
                        score += 30_000
                    elif petrel_energy_bridge and ARIANA in search_option_ids:
                        score -= 1_400
                    elif petrel_attack_bridge:
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
                    if alakazam_petrel_bridge:
                        score += 46_000
                    score += 7_200 if petrel_attack_bridge else (1_100 if needs_setup else 520)
                    if sakaki_pressure:
                        score += 90_000
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
                elif identifier == TEAM_ROCKET_ENERGY:
                    if energy_dig_needed:
                        score = 260_000 if counts.get(TEAM_ROCKET_ENERGY, 0) <= 0 else 42_000
                    else:
                        score += 1_600
                    if sakaki_pressure and counts.get(TEAM_ROCKET_ENERGY, 0) <= 0:
                        score += 58_000
                elif identifier == IGNITION_ENERGY:
                    if energy_dig_needed:
                        score = 120_000 if counts.get(IGNITION_ENERGY, 0) <= 0 else 18_000
                    else:
                        score += 900
                    if sakaki_pressure and counts.get(TEAM_ROCKET_ENERGY, 0) > 0 and counts.get(IGNITION_ENERGY, 0) <= 0:
                        score += 18_000
                elif identifier == GIOVANNI:
                    score = giovanni_fuel_search_score()
                elif identifier == ARCHER:
                    score = _apollo_search_score(observation)
                elif identifier == NIGHT_STRETCHER:
                    if alakazam_articuno_recovery:
                        score += 300_000
                    score += max(0, _best_night_stretcher_target_score(observation) // 60)
                    if sakaki_pressure:
                        score += 54_000
            elif effect == MURKROW:
                if identifier not in ROCKET_SUPPORTERS:
                    score = -100_000
                else:
                    score = _tempt_supporter_target_score(player, identifier)
            elif effect == ROTO_STICK:
                if identifier == PROTON and dragapult_articuno_needed:
                    score += 260_000
                elif identifier == PETREL and alakazam_articuno_petrel_recovery:
                    score += 260_000
                elif identifier in SUPPORTER_CARD_IDS:
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
                    if dragapult_articuno_needed:
                        score += 220_000
                    elif proton_opening_allowed:
                        score += 1_100
                    else:
                        score -= _policy_rule_number("preferProtonWhenSetupIncomplete", "settledSearchPenalty", 50_000)
                elif identifier == PETREL:
                    score += 18_000 if petrel_energy_bridge else 800
                elif identifier == GIOVANNI:
                    score = giovanni_fuel_search_score()
                elif identifier == ARCHER:
                    score = _apollo_search_score(observation)
            elif effect == NIGHT_STRETCHER:
                score = _night_stretcher_target_score(observation, option_card)
            if effect in (POKEGEAR, TEAM_ROCKET_TRANSCEIVER, PETREL, MURKROW, ROTO_STICK, MIRACLE_HEADSET) and identifier in (ARIANA, ARCHER):
                energy_dig_adjustment = _supporter_energy_dig_score_adjustment(current, player, identifier, apollo_from_hand=False)
                if energy_dig_adjustment > 0 and identifier in hand_ids:
                    energy_dig_adjustment = 0
                score += energy_dig_adjustment
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
    mulligan_bonus_draw = _choose_mulligan_bonus_draw(observation, options, max_count)
    if mulligan_bonus_draw is not None:
        return mulligan_bonus_draw
    promotion_active = _choose_promotion_active_card(observation, options, max_count)
    if promotion_active is not None:
        return promotion_active
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
    deck_ids = _deck_card_ids_for_policy(player)
    active_ids = [_card_id(card) for card in _iter_cards(_read(player, "active", []))]
    active_card = _top_card(_read(player, "active", []))
    hand_count = _hand_count(player)
    hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    hand_energy = _count_cards(player, ("hand",), lambda card_id: card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY))
    deck_energy = _deck_card_count_for_policy(player, lambda card_id: card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY))
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
    active_murkrow = any(card_id == MURKROW for card_id in active_ids)
    active_porygon2 = any(card_id == PORYGON2 for card_id in active_ids)
    active_murkrow_can_evolve = _active_murkrow_can_evolve_to_honchkrow(observation)
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
    first_proton_pending = _proton_not_in_discard_or_trash(player)
    first_proton_search_pending = first_proton_pending and PROTON not in hand_ids and not supporter_played
    early_turn = _safe_int(_read(current, "turn"), 0) <= _policy_rule_number("preferProtonWhenSetupIncomplete", "earlyTurnMax", 4)
    opponent_active_hp = _remaining_hp(_opponent_active_card(observation))
    draw_supporter_compression_available = _has_ariana_compression_option(observation)
    energy_murkrow_needs_honchkrow = _energy_prepared_murkrow_needs_honchkrow(player, hand_ids, deck_ids)
    energy_dig_needed = _needs_ariana_energy_dig(player)
    petrel_energy_bridge = energy_dig_needed and TEAM_ROCKET_ENERGY in deck_ids and TEAM_ROCKET_ENERGY not in hand_ids
    needs_honchkrow_from_pad, needs_porygon2_from_pad, active_pad_evolution = _poke_pad_evolution_attack_need(player, hand_ids)
    porygon_development_allowed = _porygon_development_allowed(player)
    petrel_attack_bridge = (
        energy_murkrow_needs_honchkrow
        or needs_honchkrow_from_pad
        or needs_porygon2_from_pad
        or active_pad_evolution
        or _alakazam_petrel_poke_pad_bridge_needed(observation)
    )
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
        and draw_supporter_compression_available
    )
    apollo_compression_needed = (
        ARCHER in hand_ids
        and not supporter_played
        and _policy_rule_enabled("compressBeforeAriana")
        and draw_supporter_compression_available
        and _apollo_reset_score(observation, assume_legal=True, apollo_from_hand=True) > 0
    )
    draw_supporter_compression_needed = ariana_compression_needed or apollo_compression_needed
    factory_pending_after_supporter = supporter_played and not stadium_played and _factory_option_available(observation)
    factory_before_ariana_score = _factory_before_ariana_score(observation)
    only_ariana_rocket_feather_fuel = _only_ariana_rocket_feather_fuel(player)
    non_ko_refill_supporter = _non_ko_refill_supporter_to_preserve(player)
    hop_dodge_protected_active = _opponent_active_has_hop_dodge_protection(observation)

    if option_type in (14, "end"):
        return -20_000

    if _alakazam_lock_strategy_active(observation):
        if option_type in (9, "evolve"):
            return -1_000_000
        if option_type in (13, "attack") and attack_id not in RULE_ONLY_MURKROW_KO_ATTACKS:
            return -1_000_000
        if option_type in (8, "attach") and target_id != MURKROW:
            return -1_000_000

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
        if hop_dodge_protected_active:
            if attack_id == ROCKET_FEATHER_ATTACK:
                return -64_000
            if attack_id in PORYGON2_R_COMMAND_ATTACKS:
                return -42_000
            if attack_id == TAUNT_ATTACK:
                return -28_000
            if attack_id == MURKROW_TEMPT_ATTACK:
                return 1_200
            return -36_000
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
            preserve_refill_fuel = (
                non_ko_refill_supporter is not None
                and _non_ko_rocket_feather_fuel_count(player, 1) <= 0
                and not can_ko
            )
            if preserve_refill_fuel:
                score = min(score, -24_000)
            non_ko_fuel_count = _non_ko_rocket_feather_split_fuel_count(
                player,
                opponent_active_hp,
                damage_per_supporter,
                hand_supporters,
            )
            damage_after_attack = max(0, opponent_active_hp - damage_per_supporter * max(1, non_ko_fuel_count))
            two_turn_progress = damage_after_attack <= damage_per_supporter * 2 or damage_after_attack <= max(60, opponent_active_hp // 2)
            if two_turn_progress and not preserve_refill_fuel:
                score += _policy_rule_number("avoidNonKoRocketFeatherIntoHealing", "twoTurnKoProgressBonus", 11_800)
            if active_has_ignition_energy and hand_supporters > 0 and not factory_pending_after_supporter and not preserve_refill_fuel:
                score = max(score, _policy_rule_number("avoidNonKoRocketFeatherIntoHealing", "ignitionTempoFloor", 52_000))
            if hand_supporters >= 2 and not factory_pending_after_supporter:
                score = max(score, _policy_rule_number("avoidNonKoRocketFeatherIntoHealing", "safeNonKoAttackFloor", 18_000) + min(hand_supporters, 3) * 900)
            if urgent_attack_window and not factory_pending_after_supporter and not preserve_refill_fuel:
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
            damage = _damage_after_type_modifier(
                discard_supporters * _policy_rule_number("porygon2RCommandFallback", "damagePerTrashRocketSupporter", 20),
                _opponent_active_card(observation),
                "colorless",
            )
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
            porygon2_can_ko = opponent_active_hp > 0 and damage >= opponent_active_hp
            late_trash_bonus = _porygon2_late_trash_bonus(discard_supporters, damage, porygon2_can_ko)
            if porygon2_can_ko:
                score = (
                    _policy_rule_number("porygon2RCommandFallback", "koScore", 14_800) +
                    min(discard_supporters, 6) *
                    _policy_rule_number("porygon2RCommandFallback", "koPerSupporterScore", 250)
                )
                score += porygon2_bridge_bonus + late_trash_bonus
                score += _porygon2_endgame_r_command_ko_bonus(player, porygon2_can_ko)
                return max(score, planned_score)
            score = (
                _policy_rule_number("porygon2RCommandFallback", "nonKoBaseScore", 6_200) +
                min(discard_supporters, 6) *
                _policy_rule_number("porygon2RCommandFallback", "nonKoPerSupporterScore", 520)
            )
            score += min(damage, 180) * 18
            score += porygon2_bridge_bonus + late_trash_bonus
            if deck_count <= _policy_rule_number("porygon2RCommandFallback", "lowDeckThreshold", 10):
                score -= _policy_rule_number("porygon2RCommandFallback", "lowDeckPenalty", 2_800)
            if urgent_attack_window and damage >= 40:
                attack_floor = 5_400 + min(discard_supporters, 6) * 550
                if discard_supporters >= _porygon2_late_trash_threshold():
                    attack_floor += _policy_rule_number("porygon2RCommandFallback", "lateTrashAttackBonus", 36_000) // 2
                    attack_floor += max(0, discard_supporters - _porygon2_late_trash_threshold()) * _policy_rule_number(
                        "porygon2RCommandFallback",
                        "lateTrashPerExtraSupporterScore",
                        1_400,
                    )
                if opponent_prizes <= 1:
                    attack_floor += 3_200
                if self_prizes <= 2:
                    attack_floor += 2_000
                score = max(score, attack_floor)
            return max(score, planned_score)
        if attack_id == TAUNT_ATTACK:
            if active_murkrow and active_murkrow_can_evolve:
                return -90_000
            if active_murkrow:
                taunt_damage = _murkrow_taunt_damage(_opponent_active_card(observation))
                if opponent_active_hp > 0 and taunt_damage >= opponent_active_hp:
                    return 86_000 + min(taunt_damage, 120) * 30
                threat = _opponent_active_attack_threat(observation)
                if threat <= 0:
                    return 1_200
                return min(6_000, 1_800 + threat * 10)
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
            if active_murkrow and active_murkrow_can_evolve:
                return -90_000
            if attack_id == MURKROW_TEMPT_ATTACK:
                score = 11_500 if active_murkrow else 3_800
                if setup_incomplete:
                    score += 1_100
                if energy_dig_needed:
                    score += 1_400
                if deck_count <= 4:
                    score -= 1_000
                return score
            if active_murkrow:
                taunt_damage = _murkrow_taunt_damage(_opponent_active_card(observation))
                if opponent_active_hp > 0 and taunt_damage >= opponent_active_hp:
                    return 82_000 + min(taunt_damage, 120) * 28
                return 1_600
            return -120_000
        return 2_200

    if option_type in (9, "evolve"):
        if seed_guard_has_basic and _needs_seed_out_bench_guard(player) and not (turn_plan is not None and turn_plan.game_end):
            return -42_000
        if (
            turn_plan is not None
            and turn_plan.needs_evolution
            and identifier == turn_plan.evolution_id
            and _option_targets_plan_attacker(target, turn_plan)
        ):
            return _policy_rule_number("donkrowTurnPlan", "evolutionBridgeActionScore", 160_000) + max(0, turn_plan.score // 4)
        if draw_supporter_compression_needed and _is_ariana_compression_option(observation, option):
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
            if discard_supporters >= _porygon2_late_trash_threshold():
                score = 58_000 if target_id == PORYGON else 28_000
                if target_is_active:
                    score += 24_000
                if target_energy_cards > 0 or IGNITION_ENERGY in hand_ids:
                    score += 18_000
                return score
            if _honchkrow_chain_available(player, hand_ids, deck_ids) and not _only_porygon_board(player):
                return 2_400
            if _only_porygon_board(player):
                return 14_800 if IGNITION_ENERGY in [_card_id(card) for card in _iter_cards(_read(player, "hand", []))] else 11_400
            if target_id == PORYGON and target_is_active and IGNITION_ENERGY in hand_ids and not _honchkrow_chain_available(player, hand_ids, deck_ids):
                return 24_000
            return 11_200 if IGNITION_ENERGY in [_card_id(card) for card in _iter_cards(_read(player, "hand", []))] else 8_600
        return 4_000

    if option_type in (8, "attach"):
        if identifier == TEAM_ROCKET_ENERGY:
            opening_attach_score = _opening_turn_team_rocket_energy_score(
                current,
                player,
                your_index,
                hand_ids,
                target_id,
                target_is_active,
                target_is_bench,
                target_energy_cards,
                target_has_rocket_energy,
            )
            if opening_attach_score is not None:
                return opening_attach_score
        if identifier == IGNITION_ENERGY and hop_dodge_protected_active:
            if target_id in (HONCHKROW, PORYGON2) and (
                target_is_active
                or has_switch_option
                or (turn_plan is not None and _option_targets_plan_attacker(target, turn_plan))
            ):
                return min(_policy_rule_number("avoidIgnitionWaste", "forbiddenTargetScore", -16_000), -32_000)
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
                    if (
                        turn_plan.attacker_id == HONCHKROW
                        and turn_plan.attack_id == ROCKET_FEATHER_ATTACK
                        and not turn_plan.can_ko
                        and only_ariana_rocket_feather_fuel
                    ):
                        return min(_policy_rule_number("avoidIgnitionWaste", "forbiddenTargetScore", -16_000), -28_000)
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
            if target_id == MURKROW:
                return _policy_rule_number("avoidIgnitionWaste", "forbiddenTargetScore", -16_000)
            if target_id == HONCHKROW:
                if not _policy_rule_bool("avoidIgnitionWaste", "allowImmediateIgnitionOnHonchkrow", True):
                    return _policy_rule_number("avoidIgnitionWaste", "forbiddenTargetScore", -16_000)
                if target_energy_cards >= 1 or has_rocket_feather_action:
                    return _policy_rule_number("avoidIgnitionWaste", "forbiddenTargetScore", -16_000)
                if not _rule_rocket_feather_ko_reachable_from_hand(observation):
                    return _policy_rule_number("avoidIgnitionWaste", "forbiddenTargetScore", -16_000)
                if TEAM_ROCKET_ENERGY in hand_ids:
                    return 600
                if hand_supporters <= 0:
                    return 700
                if not target_is_active and not has_switch_option:
                    return _policy_rule_number("avoidIgnitionWaste", "benchWithoutSwitchPenalty", -15_000)
                score = _policy_rule_number("avoidIgnitionWaste", "honchkrowActiveAttackScore", 15_800) if target_is_active else _policy_rule_number("avoidIgnitionWaste", "honchkrowSwitchableAttackScore", 10_200)
                required_for_ko = max(1, (opponent_active_hp + 59) // 60) if opponent_active_hp > 0 else 1
                if only_ariana_rocket_feather_fuel and hand_supporters < required_for_ko:
                    return min(_policy_rule_number("avoidIgnitionWaste", "forbiddenTargetScore", -16_000), -28_000)
                if hand_supporters >= required_for_ko:
                    score += 9_500
                score += min(hand_supporters, 4) * 850
                return score
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
        if draw_supporter_compression_needed and _is_ariana_compression_option(observation, option):
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
            if discard_supporters >= _porygon2_late_trash_threshold():
                return 42_000 if porygon_lines > 0 else 8_000
            if _honchkrow_chain_available(player, hand_ids, deck_ids) and not _only_porygon_board(player):
                return 600
            if _only_porygon_board(player):
                return 8_800 if porygon_lines > 0 else 1_200
            return 5_800 if porygon_lines > 0 else 800
        if identifier == PROTON:
            if supporter_played:
                return -7_500
            if first_proton_pending:
                return 300_000
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
            if petrel_energy_bridge:
                score += 34_000
            if petrel_attack_bridge:
                score += 8_600
            if petrel_factory_bridge:
                score += 6_200
            if rocket_feather_fuel_need and active_honchkrow:
                score += 1_700
            if energy_murkrow_needs_honchkrow:
                score += 2_900
            if not petrel_energy_bridge and not petrel_attack_bridge and not petrel_factory_bridge:
                score -= 6_800
            if energy_dig_needed:
                score += 13_000 if petrel_energy_bridge else -(7_800 if ARIANA in hand_ids else 2_800)
            if setup_incomplete and (PROTON in hand_ids or TEAM_ROCKET_TRANSCEIVER in hand_ids or POKE_PAD in hand_ids):
                score -= 3_200
            if hand_count <= 5 and hand_energy <= 0 and not petrel_energy_bridge:
                score -= 2_400
            return score
        if identifier == TEAM_ROCKET_TRANSCEIVER:
            if first_proton_search_pending:
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
            if first_proton_search_pending:
                return 235_000
            return 8_800 + (3_900 if rocket_feather_fuel_need else 0) + (2_200 if energy_murkrow_needs_honchkrow else 0) + (
                _policy_rule_number("preferArianaEnergyDig", "pokegearArianaBonus", 5_200) if energy_dig_needed else 0
            )
        if identifier == FACTORY:
            if stadium_played or not supporter_played:
                return -6_500
            return 72_000
        if identifier == ROTO_STICK:
            if first_proton_search_pending:
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
                if draw_supporter_compression_available:
                    score -= _policy_rule_number("compressBeforeAriana", "compressionPenalty", 8_500)
            score += _supporter_energy_dig_score_adjustment(current, player, ARIANA, apollo_from_hand=True)
            if proton_opening_allowed and early_turn and (PROTON in hand_ids or TEAM_ROCKET_TRANSCEIVER in hand_ids or POKE_PAD in hand_ids):
                score -= 5_400
            return score
        if identifier == GIOVANNI:
            if supporter_played:
                return -7_500
            sakaki_ko_score = _sakaki_prize_race_ko_score(observation, giovanni_from_hand=True)
            if sakaki_ko_score is not None:
                return sakaki_ko_score
            return min(_policy_rule_number("sakakiRequiresKo", "nonKoScore", -9_000), -72_000)
        if identifier == ARCHER:
            if supporter_played:
                return -7_500
            if apollo_compression_needed:
                return -18_000
            apollo_score = _apollo_reset_score(observation, assume_legal=True, apollo_from_hand=True)
            energy_dig_adjustment = _supporter_energy_dig_score_adjustment(current, player, ARCHER, apollo_from_hand=True)
            if apollo_score > 0:
                if energy_dig_needed and ARIANA in hand_ids and energy_dig_adjustment <= 0:
                    apollo_score -= 6_000
                return apollo_score + energy_dig_adjustment
            return apollo_score + energy_dig_adjustment if energy_dig_adjustment > 0 else apollo_score
        return 500

    if option_type in (10, "ability"):
        if identifier == FACTORY:
            if stadium_played or not supporter_played:
                return -6_500
            return 70_000
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


def _pre_support_board_development_score(observation, option, turn_plan=None, draw_reset_supporter_pending=False):
    current, _, player = _current_player(observation)
    if _supporter_played_this_turn(current, player):
        return None
    if turn_plan is not None and turn_plan.game_end:
        return None

    option_type = _option_type(option)
    card = _card_from_option(observation, option)
    identifier = _card_id(card)
    if identifier is None:
        return None

    hand_ids = [_card_id(hand_card) for hand_card in _iter_cards(_read(player, "hand", []))]
    deck_ids = _deck_card_ids_for_policy(player)
    field_top_ids = [_card_id(field_card) for field_card in _field_top_cards(player)]
    discard_supporters = _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    bench_count = _bench_top_count(player)
    free_bench = bench_count < _bench_limit(player)

    if option_type in (9, "evolve"):
        main_score = _main_action_score(observation, option, turn_plan)
        if main_score <= 0:
            return None
        if identifier == HONCHKROW:
            return 180_000 + min(main_score, 60_000)
        if identifier == PORYGON2 and _porygon_development_allowed(player):
            if discard_supporters >= _porygon2_late_trash_threshold():
                return 170_000 + min(main_score, 80_000)
            if _honchkrow_chain_available(player, hand_ids, deck_ids) and not _only_porygon_board(player):
                return None
            return 95_000 + min(main_score, 40_000)
        return None

    if option_type not in (7, "play"):
        return None

    if identifier == MURKROW and free_bench:
        line_count = field_top_ids.count(MURKROW) + field_top_ids.count(HONCHKROW)
        if line_count >= 3:
            return None
        return 135_000 - line_count * 8_000

    if identifier == PORYGON and free_bench:
        if PORYGON in field_top_ids or PORYGON2 in field_top_ids:
            return None
        return 92_000

    if identifier == POKE_PAD:
        target_score = _best_poke_pad_target_score(observation)
        if draw_reset_supporter_pending:
            return 82_000 + max(0, min(target_score, 180_000) // 12)
        if target_score < 75_000:
            return None
        return 78_000 + min(target_score // 10, 28_000)

    if identifier in (TEAM_ROCKET_TRANSCEIVER, POKEGEAR, ROTO_STICK):
        if draw_reset_supporter_pending:
            base_scores = {
                TEAM_ROCKET_TRANSCEIVER: 86_000,
                POKEGEAR: 78_000,
                ROTO_STICK: 74_000,
            }
            return base_scores.get(identifier, 72_000)

    if identifier == NIGHT_STRETCHER:
        target_score = _best_night_stretcher_target_score(observation)
        immediate_score = _night_stretcher_immediate_compression_score(observation)
        if draw_reset_supporter_pending and immediate_score >= 75_000:
            return 76_000 + min(immediate_score // 10, 30_000)
        if target_score < 75_000:
            return None
        return 72_000 + min(target_score // 10, 26_000)

    return None


def _pre_support_board_development_option_index(observation, options, turn_plan=None):
    current, _, player = _current_player(observation)
    if _supporter_played_this_turn(current, player):
        return None
    draw_reset_supporter_pending = any(
        _option_type(option) in (7, "play") and _card_id(_card_from_option(observation, option)) in DRAW_RESET_SUPPORTERS
        for option in options
    )
    has_supporter_option = any(
        _option_type(option) in (7, "play") and _card_id(_card_from_option(observation, option)) in SUPPORTER_CARD_IDS
        for option in options
    )
    if not has_supporter_option:
        return None

    best = None
    for option_index, option in enumerate(options):
        score = _pre_support_board_development_score(observation, option, turn_plan, draw_reset_supporter_pending)
        if score is None:
            continue
        if best is None or score > best[0] or (score == best[0] and option_index < best[1]):
            best = (score, option_index)
    return best[1] if best is not None else None


def _pre_attack_action_option_index(observation, options, scored):
    current, _, player = _current_player(observation)
    supporter_available = not _supporter_played_this_turn(current, player)

    best = None
    for score, option_index in scored:
        option = options[option_index]
        option_type = _option_type(option)
        if option_type in (13, "attack", 14, "end", 12, "retreat", "switch", "promote"):
            continue

        identifier = _card_id(_card_from_option(observation, option))
        if option_type in (7, "play") and identifier in SUPPORTER_CARD_IDS:
            if not supporter_available:
                continue
            if identifier in (PROTON, GIOVANNI) and score <= 0:
                continue
            if identifier == ARCHER and score < 0:
                continue
            if identifier in (ARIANA, PETREL) and score < -6_500:
                continue
        elif score <= 0:
            continue

        priority = 0
        if option_type in (9, "evolve"):
            priority = 7
        elif option_type in (7, "play") and identifier in (ARIANA, PETREL, ARCHER):
            priority = 6
        elif option_type in (7, "play") and identifier in (POKE_PAD, NIGHT_STRETCHER, TEAM_ROCKET_TRANSCEIVER, POKEGEAR, ROTO_STICK):
            priority = 5
        elif identifier == FACTORY:
            priority = 4
        elif option_type in (8, "attach"):
            priority = 3
        elif option_type in (7, "play", 10, "ability"):
            priority = 2

        candidate = (priority, score, -option_index, option_index)
        if best is None or candidate > best:
            best = candidate

    return best[3] if best is not None else None


def _sakaki_support_phase_option_index(observation, options, scored, turn_plan):
    current, _, player = _current_player(observation)
    if _supporter_played_this_turn(current, player):
        return None

    hand_ids = [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]
    if GIOVANNI not in hand_ids:
        return None

    giovanni_option_index = None
    for option_index, option in enumerate(options):
        if _option_type(option) in (7, "play") and _card_id(_card_from_option(observation, option)) == GIOVANNI:
            giovanni_option_index = option_index
            break

    if giovanni_option_index is not None and _sakaki_prize_race_ko_score(observation, giovanni_from_hand=True) is not None:
        return giovanni_option_index

    if not _sakaki_bench_prize_pressure(observation):
        return None

    best_overall = scored[0] if scored else None
    if best_overall is not None:
        best_option = options[best_overall[1]]
        if _option_type(best_option) in (13, "attack") and turn_plan is not None and turn_plan.game_end:
            return None

    candidate = None
    for score, option_index in scored:
        option = options[option_index]
        option_type = _option_type(option)
        identifier = _card_id(_card_from_option(observation, option))
        target = _target_card_from_option(observation, option)
        target_id = _card_id(target)
        priority = 0

        if option_type in (9, "evolve") and identifier in (HONCHKROW, PORYGON2):
            priority = 8
        elif option_type in (7, "play") and identifier in (POKE_PAD, NIGHT_STRETCHER):
            priority = 7
        elif option_type in (7, "play") and identifier == MURKROW:
            priority = 6
        elif option_type in (8, "attach") and target_id in (HONCHKROW, MURKROW, PORYGON2):
            priority = 5
        elif option_type in (7, "play") and identifier == PETREL and score > 0:
            priority = 4

        if priority <= 0 or score <= 0:
            continue

        ranked = (priority, score, -option_index, option_index)
        if candidate is None or ranked > candidate:
            candidate = ranked

    return candidate[3] if candidate is not None else None


def _tempo_attack_fallback_option_index(observation, options, scored, turn_plan):
    _, _, player = _current_player(observation)
    if _opponent_active_has_hop_dodge_protection(observation):
        return None
    if _needs_seed_out_bench_guard(player) and _has_basic_play_option(observation):
        return None
    if turn_plan is not None and turn_plan.game_end:
        return None

    hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    for score, option_index in scored:
        option = options[option_index]
        if _option_type(option) not in (13, "attack"):
            continue
        attack_id = _attack_id_from_option(option)
        if attack_id == MURKROW_TEMPT_ATTACK and hand_supporters > 0:
            continue
        if score <= -900_000:
            continue
        return option_index
    return None


def _turn_plan_conversion_option_index(observation, options, turn_plan):
    if turn_plan is None or not turn_plan.can_ko:
        return None

    best_evolution = None
    best_attach = None
    best_switch = None
    for option_index, option in enumerate(options):
        option_type = _option_type(option)
        target = _target_card_from_option(observation, option)
        if (
            turn_plan.needs_evolution
            and option_type in (9, "evolve")
            and _rule_option_id(observation, option) == turn_plan.evolution_id
            and _option_targets_plan_attacker(target, turn_plan)
        ):
            score = _main_action_score(observation, option, turn_plan)
            if best_evolution is None or score > best_evolution[0]:
                best_evolution = (score, option_index)
            continue

        if (
            turn_plan.needs_energy
            and option_type in (8, "attach")
            and _option_targets_plan_attacker(target, turn_plan)
        ):
            score = _main_action_score(observation, option, turn_plan)
            if score > 0 and (best_attach is None or score > best_attach[0]):
                best_attach = (score, option_index)
            continue

        if turn_plan.needs_switch and option_type in (12, "retreat", "switch", "promote"):
            promotion_card = _promotion_card_from_option(observation, option)
            if _switch_option_targets_plan_attacker(option, target, promotion_card, turn_plan):
                score = _main_action_score(observation, option, turn_plan)
                if score > 0 and (best_switch is None or score > best_switch[0]):
                    best_switch = (score, option_index)

    for candidate in (best_evolution, best_attach, best_switch):
        if candidate is not None:
            return candidate[1]

    if (
        turn_plan.attacker_index == 0
        and not turn_plan.needs_evolution
        and not turn_plan.needs_energy
        and not turn_plan.needs_switch
        and (turn_plan.game_end or not turn_plan.seed_guard_blocked)
    ):
        return _rule_find_attack_option(options, turn_plan.attack_id)
    return None


def _choose_donkrow_main_action(observation, options):
    select = _read(observation, "select", {})
    if _select_context(select) not in (0, "main"):
        return None

    current, _, player = _current_player(observation)
    supporter_played = _supporter_played_this_turn(current, player)
    turn_plan = _build_donkrow_turn_plan(observation)

    ko_conversion_index = _turn_plan_conversion_option_index(observation, options, turn_plan)
    if ko_conversion_index is not None:
        return ko_conversion_index

    if supporter_played:
        for option_index, option in enumerate(options):
            if _card_id(_card_from_option(observation, option)) == FACTORY and _option_type(option) in (10, "ability"):
                return option_index

    pre_support_development_index = _pre_support_board_development_option_index(observation, options, turn_plan)
    if pre_support_development_index is not None:
        return pre_support_development_index

    factory_before_ariana_index = _factory_before_ariana_option_index(observation, options)
    if factory_before_ariana_index is not None:
        return factory_before_ariana_index

    has_draw_reset_supporter_option = any(
        _card_id(_card_from_option(observation, option)) in DRAW_RESET_SUPPORTERS
        for option in options
    )

    if has_draw_reset_supporter_option:
        for option_index, option in enumerate(options):
            identifier = _card_id(_card_from_option(observation, option))
            option_type = _option_type(option)
            if option_type in (9, "evolve") and identifier in (HONCHKROW, PORYGON2):
                return option_index

        for option_index, option in enumerate(options):
            if _option_type(option) in (8, "attach"):
                target = _target_card_from_option(observation, option)
                target_id = _card_id(target)
                score = _main_action_score(observation, option, turn_plan)
                if score <= 0:
                    continue
                if target_id in (HONCHKROW, MURKROW):
                    return option_index
                if target_id == PORYGON2 and turn_plan is not None and _option_targets_plan_attacker(target, turn_plan):
                    return option_index

    scored = []
    for option_index, option in enumerate(options):
        scored.append((_main_action_score(observation, option, turn_plan), option_index))
    if not scored:
        return None

    scored.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_index = scored[0]
    best_option = options[best_index]
    best_identifier = _card_id(_card_from_option(observation, best_option))
    best_option_type = _option_type(best_option)

    sakaki_support_phase_index = _sakaki_support_phase_option_index(observation, options, scored, turn_plan)
    if sakaki_support_phase_index is not None:
        return sakaki_support_phase_index

    if best_option_type in (13, "attack") and not (turn_plan is not None and turn_plan.game_end):
        pre_attack_index = _pre_attack_action_option_index(observation, options, scored)
        if pre_attack_index is not None:
            return pre_attack_index

    if best_score > 0 and best_identifier == ARCHER and best_option_type in (7, "play"):
        for option_index, option in enumerate(options):
            identifier = _card_id(_card_from_option(observation, option))
            option_type = _option_type(option)
            if option_type in (9, "evolve") and identifier in (HONCHKROW, PORYGON2):
                if _main_action_score(observation, option, turn_plan) > 0:
                    return option_index

        for option_index, option in enumerate(options):
            if _option_type(option) in (8, "attach"):
                target = _target_card_from_option(observation, option)
                target_id = _card_id(target)
                score = _main_action_score(observation, option, turn_plan)
                if score <= 0:
                    continue
                if target_id in (HONCHKROW, MURKROW):
                    return option_index
                if target_id == PORYGON2 and turn_plan is not None and _option_targets_plan_attacker(target, turn_plan):
                    return option_index

    if best_score > 0:
        return best_index
    if best_score == 0:
        return None
    tempo_attack_index = _tempo_attack_fallback_option_index(observation, options, scored, turn_plan)
    if tempo_attack_index is not None:
        return tempo_attack_index
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
        _reset_public_knowledge()
        return MY_DECK

    _remember_public_information(obs_dict)
    _remember_hop_phantump_dodge(obs_dict)
    _remember_observed_attacks(obs_dict)

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


_scored_agent_impl = agent


# BEGIN RULE_ONLY_AGENT
# This experimental variant keeps the latest parser/card helpers from the main
# Donkrow submission, but replaces the main-phase decision with a strict
# phase-ordered rule engine. It is intentionally less nuanced than the scored
# policy so we can inspect the CPU's skeleton without score interactions.

ABC_LAB_TRACE_ENABLED = True

RULE_ONLY_SUPPORTER_AFTER_LANCE = (ARIANA, PETREL, ARCHER, GIOVANNI, PROTON)
RULE_ONLY_FUEL_DISCARD_ORDER = (PROTON, GIOVANNI, ARCHER, PETREL, ARIANA)
RULE_ONLY_MIRACLE_RECOVERY_ORDER = (ARIANA, ARCHER, PETREL, GIOVANNI, PROTON)
RULE_ONLY_CARD_NAMES_JA = {
    MURKROW: "ロケット団のヤミカラス",
    HONCHKROW: "ロケット団のドンカラス",
    PORYGON: "ロケット団のポリゴン",
    PORYGON2: "ロケット団のポリゴン2",
    ARTICUNO: "ロケット団のフリーザー",
    DREEPY: "ドラメシヤ",
    DRAKLOAK: "ドロンチ",
    DRAGAPULT_EX: "ドラパルトex",
    TWM_ABRA: "ケーシィ",
    TWM_ALAKAZAM: "フーディン",
    ABRA: "ケーシィ",
    KADABRA: "ユンゲラー",
    ALAKAZAM: "フーディン",
    TEAM_ROCKET_ENERGY: "ロケット団エネルギー",
    IGNITION_ENERGY: "イグニッションエネルギー",
    POKEGEAR: "ポケギア",
    TEAM_ROCKET_TRANSCEIVER: "ロケット団のレシーバー",
    ROTO_STICK: "ロトスティック",
    POKE_PAD: "ポケパッド",
    NIGHT_STRETCHER: "夜のタンカ",
    MIRACLE_HEADSET: "ミラクルインカム",
    PROTON: "ランス",
    PETREL: "ラムダ",
    GIOVANNI: "サカキ",
    ARCHER: "アポロ",
    ARIANA: "アテナ",
    FACTORY: "ロケット団のファクトリー",
}
RULE_ONLY_ATTACK_NAMES_JA = {
    ROCKET_FEATHER_ATTACK: "ロケットフェザー",
    TAUNT_ATTACK: "いちゃもん",
    MURKROW_TEMPT_ATTACK: "たぶらかす",
    MURKROW_SECONDARY_ATTACK: "ヤミカラスの補助ワザ",
    670: "Rコマンド",
}
RULE_ONLY_PHASES_JA = (
    ("1", "進化", "ドンカラス、ポリゴン2に進化できるなら先に進化する。ただしドラパルトLineが相手盤面に見えた対面では、攻撃が確定する時だけ進化する。"),
    ("2", "たね展開", "通常はヤミカラス、ポリゴンを出す。ドラパルト対面ではフリーザー未設置なら最優先し、設置後にヤミカラス2体、ポリゴンの順で見る。"),
    ("2.5", "初動R団エネルギー", "先攻/後攻1ターン目のロケット団エネルギー手張りルールだけ、サポート前準備より先に確認する。"),
    ("3", "サポート前準備", "サカキ非即時確認、サポート前手張り、サポート確定時のファクトリー先置き、ポケパッド、夜のタンカ、レシーバー、ポケギア、ロトスティックを見る。攻撃はここでは実行せず、KOを壊す行動だけガードする。"),
    ("4", "サポート使用", "攻撃で番を終える前に、初回ランス条件、サカキ確定KO/ボクレー回避、エネルギー探索確率、アテナ、アテナ薄ドロー時のアポロ、ラムダ接続、低山札/補充アポロの順に見る。KO担保を壊すサポートは止める。"),
    ("5", "サポート後ファクトリー/ポケパッド", "サポート使用後、mainに戻った時に、打点不足ならファクトリー先、攻撃条件不足ならロトスティック→ポケギア→レシーバーで山札のサポートを抜いてからファクトリーを見る。"),
    ("6", "エネルギー手張り", "攻撃に必要な先、または後続アタッカーへ手張りする。"),
    ("7", "攻撃", "ロケットフェザーとRコマンドを必要打点、サポート消費、補充見込み、トラッシュ打点で比較する。固定ルール後にフローガードを通過した同一段階候補だけを軽く再評価し、攻撃接続しない逃げや実益なしグッズ消費は保護/抑制する。"),
    ("8", "ヤミカラス緊急ワザ", "いちゃもん系でKOできるなら使う。KOできない時はサポートなしならたぶらかす。"),
    ("9", "終了", "他に選べるルールがない場合だけ番を終える。"),
)
RULE_ONLY_DETAIL_PHASES_JA = (
    ("0-1", "初期デッキ提出", "select がない", "60枚の deck.csv 相当を返す"),
    ("0-2", "メインフェーズ判定", "select context が main", "_rule_choose_main_action の順番で選ぶ"),
    ("0-3", "メイン外任意選択判定", "select context が main 以外", "_rule_choose_optional_multi_select の順番で選ぶ"),
    ("0-4", "サカキ即時リーサル確認", "サポート未使用で、手札のサカキにより相手ベンチを取り、残りサイドを取り切れる", "進化・たね展開・サポート前手張りより前にサカキを使う"),
    ("1-1", "進化", "ロケット団のドンカラスへ進化できる。ドラパルト対面では前のヤミカラスが進化後に攻撃できる時だけ許可", "第1候補としてドンカラスに進化。ドラパルト対面で攻撃に直結しない進化は保留"),
    ("1-2", "進化", "ロケット団のポリゴン2へ進化できる。ドラパルト対面では前のポリゴンが進化後にRコマンドへつながる時だけ許可", "次にポリゴン2へ進化。ドラパルト対面で攻撃に直結しない進化は保留"),
    ("1-D", "ドラパルト対面進化ガード", "相手盤面にドラパルトLineが1枚以上見えており、進化してもこのターン攻撃が確定しない", "ドンカラス/ポリゴン2への進化を止め、たね展開と攻撃準備を優先"),
    ("2-1", "たね展開", "ロケット団のヤミカラスを場に出せる", "ヤミカラスを優先して出す"),
    ("2-2", "たね展開", "ロケット団のポリゴンを場に出せる", "ヤミカラスがなければポリゴンを出す"),
    ("2-D", "ドラパルト対面たね展開", "相手盤面にドラパルトLineが1枚以上見えている", "フリーザー未設置かつアクセス可能なら最優先で置く。設置後はヤミカラス2体、ポリゴンの順。フリーザー未設置で既存ポリゴンLineがある、または残りベンチ1枠なら追加ポリゴンを止める"),
    ("2-3", "初動R団エネルギー", "先攻1ターン目で前がヤミカラス、または前がヤミカラス以外でベンチにヤミカラスがいる", "該当するヤミカラスにロケット団エネルギーを貼る"),
    ("2-4", "初動R団エネルギー", "後攻1ターン目でロケット団エネルギーが2枚以上あり、ベンチにヤミカラスがいる", "ベンチのヤミカラスにロケット団エネルギーを貼る"),
    ("2-5", "初動R団エネルギー", "後攻1ターン目でロケット団エネルギーが1枚だけ、かつまだ有効なヤミカラス後続を展開できていない", "初動専用判定では温存するが、展開後は通常手張りでベンチのヤミカラス/ドンカラスへ貼る"),
    ("3-0", "サポート前リーサル確認", "手札のサカキで相手ベンチを取り、残りサイドを取り切れる", "3番グッズより前にサカキを使う。即時リーサルは0-4で先に処理する"),
    ("3-0.5", "サポート前手張り", "手張り未使用で、対象にエネルギーが付いていない。ヤミカラス/ドンカラスへロケット団エネルギー、前のドンカラスへKOや高サイド削りに直結するイグニッション、前のポリゴン/ポリゴン2へRコマンドが1サイドKOまたは2サイド以上2回圏内になるイグニッションを貼れる。フリーザーへは貼らない。対フーディンで前ポリゴンがロックを邪魔している時だけ、逃げるための前ポリゴンへのイグニッションを許可する", "サポート前に安全なエネルギーを貼る。ポリゴンはRコマンド圧力条件、または対フーディンの前縛り脱出に限ってイグニッションを許可する。フリーザーはエネルギー対象から除外する"),
    ("3-0.6", "サポート前ファクトリー設置", "この後サポートを使うことが確定していて、手札にファクトリーがあり、場にファクトリーがなく、スタジアム権利が残っている", "サポート使用前にファクトリーを置き、サポート後の2ドロー権利を確定させる"),
    ("3-0.8", "KO担保保護", "ロケットフェザーまたはRコマンドの攻撃 option があり、現在の手札/トラッシュ打点だけで相手バトル場をKOできる", "ここでは攻撃しない。攻撃は7で最後に行う。ただしアポロ/アテナ/ラムダ等でKO燃料や攻撃条件を壊す場合は止め、アテナ/アポロで必要燃料を高確率で引き直せる時だけ例外許可する"),
    ("3-1", "サポート前準備", "ポケパッドが使える。ただし初回ランスへ到達できる時は、先にランスでタネ展開を済ませる", "盤面が弱い時だけタネ確保に使う。ランス後または盤面形成後は進化先確保を優先する"),
    ("3-2", "サポート前準備", "夜のタンカが使える", "種切れ回避や進化ライン回収を先に行う"),
    ("3-3", "サポート前準備", "ロケット団のレシーバーが使える。ただし、アテナまたは初回条件を満たすランスをこのまま使える時は飛ばす", "使うべきサポートが手札にない時だけ変換する"),
    ("3-4", "サポート前準備", "ポケギアが使える。ただし、アテナまたは初回条件を満たすランスをこのまま使える時は飛ばす", "使うべきサポートが手札にない時だけサポートに触る"),
    ("3-5", "サポート前準備", "ロトスティックが使える", "アテナ/アポロ前でも保持せず、サポートまたは選択候補カードに触るために使う"),
    ("4-1", "サポート使用", "ベンチ2体未満、またはヤミカラス/ドンカラス合計2体未満で、ロケット団サポート使用履歴0枚", "ランスを使う"),
    ("4-2", "サポート使用", "バトル場の確定KOまたは高サイドへの十分な削りを基準に、後ろにより高い価値の確定KOがある時だけサカキを使う。前の2サイド以上へ2ターンKOの圧をかけられる時は、後ろ1サイドKO目的で使わない。相手のボクレー系回避で前に攻撃が通らない時は、サカキでベンチの確定KOへ逃がせるなら使う", "サカキを使う"),
    ("4-3", "サポート使用", "未手張り・エネルギー探索中で、アポロのエネルギー到達確率がアテナを0.1%以上上回る", "アポロを使う"),
    ("4-4", "サポート使用", "アテナのドロー枚数が1枚以上、またはエネルギー探索確率でアテナがアポロ以上", "アテナを使う"),
    ("4-5", "サポート使用", "ファクトリー未設置、またはラムダからポケパッドで進化/盤面形成へ触れる、またはエネルギー探索へ触れる。対フーディンでフリーザーがトラッシュにあり夜のタンカが山札にある時はラムダで復旧へつなぐ", "ラムダを使う。対フーディンのフリーザー復旧では夜のタンカを探す"),
    ("4-6", "サポート使用", "山札が4枚以下、手札が少ない、アテナの実ドローが薄い、ロケットフェザー燃料が薄い、またはエネルギー/進化札を探したい", "アポロを使い、手札リセットでエネルギー、進化、次ターン用燃料をまとめて探す"),
    ("5-0", "サポート後・攻撃条件不足時のファクトリー前圧縮", "サポート使用済みで、ファクトリーが使え、まだ攻撃 option がなく、エネルギーまたは進化先が不足し、山札にロケット団サポートが残っている", "ロトスティック、ポケギア、レシーバーの順で山札のサポートを抜き、ファクトリー2ドローのエネルギー/進化先到達率を上げる"),
    ("5-1", "サポート後ファクトリー", "サポート使用済みで、効果解決後にmainへ戻り、スタジアム権利が残り、ファクトリーが使える。ロケットフェザーの打点不足時はここを最優先し、5-0の攻撃条件不足圧縮が不要または完了している", "ファクトリーを使う"),
    ("5-1.5", "攻撃前ミラクルインカム", "ロケットフェザー攻撃 option があり、相手バトル場が2サイド以上。現在の手札燃料ではKOできないが、ミラクルインカムでトラッシュのロケット団サポートを戻せばKOできる", "攻撃前にミラクルインカムを使い、不足枚数分をアテナ、アポロ、ラムダ、サカキ、ランスの順で戻す。手札の不要サポートをロケットフェザーの燃料に回す"),
    ("5-2", "サポート後ポケパッド", "サポート使用済みで、効果解決後に main へ戻り、ポケパッドが使える", "対象があれば必ず抜く。対象がなければ対象なし選択を許容し、攻撃や手張りより前に圧縮する"),
    ("5-3", "攻撃前燃料補助", "サポート使用後、ロケットフェザー攻撃optionがあり、山札にロケット団サポートが残っている", "打点が足りない時はファクトリー後にロトスティック、ポケギア、レシーバーの順で燃料補助を見る。既にKOできる場合でも、サポート燃料や次ターン選択肢のために損のない範囲で使う"),
    ("6-1", "エネルギー手張り", "序盤かつ前のヤミカラスにロケット団エネルギーを貼れる", "前のヤミカラスに貼る"),
    ("6-2", "エネルギー手張り", "序盤かつベンチのヤミカラスにロケット団エネルギーを貼れる", "ベンチのヤミカラスに貼る"),
    ("6-3", "エネルギー手張り", "前のヤミカラス/ドンカラスにエネルギーがなく、ロケット団エネルギーを貼れる", "前のヤミカラス/ドンカラスにロケット団エネルギー"),
    ("6-4", "エネルギー手張り", "ヤミカラスにはイグニッションを貼らない。前のドンカラスにエネルギーがなく、R団エネが手札になく、ロケットフェザーKO・ミラクルインカム後の高サイドKO・相手バトル場2サイド以上への十分な削りへ直結する時だけイグニッションを許可", "ヤミカラスは禁止。ドンカラスはKO攻撃、ミラクルインカム後の高サイドKO、または2ターンKOの前進へ進める場合だけイグニッション"),
    ("6-5", "エネルギー手張り", "前のポリゴン2で、Rコマンドが相手バトル場の1サイドをKO、または2サイド以上を2回圏内にでき、イグニッションを貼れる。対フーディンで前ポリゴンがいちゃもんロックを邪魔し、逃げるための手張りが必要な時も例外許可する", "前のポリゴン2にイグニッション。対フーディン前ポリゴン脱出では前ポリゴンにイグニッション"),
    ("6-6", "エネルギー手張り", "すでに攻撃可能で、ベンチのドンカラスにロケット団エネルギーを貼れる", "後続ドンカラスを育成"),
    ("6-7", "エネルギー手張り", "すでに攻撃可能で、ベンチのヤミカラスにロケット団エネルギーを貼れる", "次のドンカラス候補を育成"),
    ("6-8", "エネルギー手張り", "攻撃不能で、ドンカラスにロケット団エネルギーを貼れる", "攻撃役候補のドンカラスを育成"),
    ("6-9", "エネルギー手張り", "攻撃不能で、ヤミカラスにロケット団エネルギーを貼れる", "次のドンカラス候補を育成"),
    ("7-1", "攻撃", "相手のホップ系回避フラグが true", "まずサカキでベンチ確定KOへ逃がせるかを確認し、無理なら攻撃を抑制"),
    ("7-2", "攻撃", "ロケットフェザーが使える", "KOを優先し、KOできない時もアテナを1枚残す。アテナがない時はアポロを1枚残し、2ターンKOへ進めるなら余剰サポートを厚めに落とす前提で攻撃として評価する"),
    ("7-3", "攻撃", "Rコマンドが使える", "トラッシュのロケット団サポート枚数による打点をロケットフェザーのコスト効率と比較して攻撃。残りサイド3以下でRコマンドでもロケットフェザーでもKOが期待できる時は、手札サポートを消費しないRコマンドKOを優先"),
    ("7-4", "攻撃", "攻撃 option が残り、進化/手張り/サポート/ファクトリーの優先行動がない", "明確な無効攻撃や種切れ回避中でなければ、番を終えず最良の攻撃を使う"),
    ("7-5", "軽量先読み補正", "固定ルールで選んだ候補を基準に、同一フロー段階のmain候補だけを軽量MCTS風に1〜2手先評価する。終了判定、KO担保保護、サカキリーサル、種切れ回避、禁止手張り、サポート前の進化・たね展開・安全手張りは上書きしない。ミラクルインカムはKO/アテナ救済/終盤取り切りに繋がる時だけ候補に残す", "固定ルール候補を実行時契約として守り、その契約内で攻撃接続、次ターン到達、ファクトリー、探索、軽量ニューラル評価を比較する"),
    ("8-1", "ヤミカラス緊急ワザ", "前がヤミカラスで、いちゃもん系ワザで相手バトル場をKOできる", "いちゃもん系ワザを使う"),
    ("8-2", "ヤミカラス緊急ワザ", "いちゃもん系でKOできず、手札にロケット団サポートがない", "たぶらかすを使う"),
    ("8-3", "ヤミカラス緊急ワザ", "いちゃもん系でKOできず、手札にロケット団サポートがある", "ヤミカラスの緊急ワザは使わない"),
    ("9-1", "終了", "番を終える選択肢がある", "番を終える"),
    ("9-2", "終了", "番を終える選択肢が見つからない", "安全フォールバックで先頭の合法選択肢"),
)
RULE_ONLY_OPTIONAL_DETAIL_PHASES_JA = (
    ("M-1", "マリガン追加ドロー", "context 38 かつ数字付き選択肢がある", "最も大きい枚数を選ぶ"),
    ("P-1", "きぜつ後の前出し", "前出し context かつ1体選択。手札にサカキがあり、攻撃役をベンチに残すと相手ベンチの2サイド以上を確定KOできる時は先に見る", "攻撃役を前に出さず、サカキ用にベンチへ残す"),
    ("P-2", "きぜつ後の前出し", "手札にドンカラスがあり、前出し候補のヤミカラスにエネルギーが付いている", "進化して攻撃役になれるヤミカラスを優先。それ以外はドンカラスを先に出す"),
    ("P-3", "きぜつ後の前出し", "通常の前出し", "ドンカラス、ポリゴン2、ヤミカラス、ポリゴンの順"),
    ("P-D", "ドラパルト対面前出し", "相手盤面にドラパルトLineが1枚以上見えている", "エネルギー付きかつダメージが乗った攻撃ラインを最優先。次にエネルギー付き攻撃ライン、ダメージあり攻撃ライン、ヤミカラス/ポリゴンLineの順。フリーザーは前に出さない"),
    ("T-1", "いちゃもんのワザ選択", "いちゃもん/taunt 系のワザ選択画面、またはヤミカラス効果のワザID選択画面", "ワザ候補として認識"),
    ("T-2", "いちゃもんのワザ選択", "複数ワザ候補がある", "同じ試合で相手が使ったワザを優先し、未知で複数なら2個目のワザを止める"),
    ("F-1", "ロケットフェザーコスト", "効果元がドンカラス", "相手バトル場の残りHPを読む。context 8 だけでは判定しない"),
    ("F-2", "ロケットフェザーコスト", "KOに必要な枚数を計算できる", "KOを逃さない範囲でアテナを1枚、アテナがない時はアポロを1枚残し、それでも足りない時だけ保護札も含めて必要枚数を捨てる"),
    ("F-3", "ロケットフェザーコスト", "任意選択でKOできず、アテナまたはアポロを残す余剰燃料がない", "捨てない"),
    ("F-4", "ロケットフェザーコスト", "KOまたは非KOの余剰燃料がある", "非KOではアテナを1枚残す。アテナがない時はアポロを1枚残し、ランス、サカキ、余剰アポロ、ラムダ、余剰アテナの順でコスト"),
    ("F-5", "ロケットフェザーコスト", "非KOでアテナまたはアポロがあり、2ターンKOへ進める削りになる", "アテナを1枚守る。アテナがない時はアポロを1枚守る。そのうえで残りのロケット団サポートは、次ターンのアテナ/アポロとRコマンド火力のために厚めに捨てる"),
    ("H-1", "ミラクルインカム選択", "相手バトル場が2サイド以上で、トラッシュのロケット団サポートを戻せばロケットフェザーKOに届く", "不足枚数だけ戻す。戻す優先度はアテナ、アポロ、ラムダ、サカキ、ランス。手札の不要サポートをコストに回す"),
    ("S-1", "サポート選択", "初回ランス条件を満たす", "ランスを選ぶ"),
    ("S-2", "サポート選択", "未手張り・エネルギー探索中で、アポロのエネルギー到達確率がアテナを0.1%以上上回る", "アポロ、アテナの順"),
    ("S-3", "サポート選択", "未手張り・エネルギー探索中で、アテナのエネルギー到達確率がアポロ以上", "アテナ、アポロの順"),
    ("S-4", "サポート選択", "ランス初回後の通常検索", "アテナ、ラムダ、アポロ、サカキ、ランスの順"),
    ("K-1", "ポケモン選択", "種切れリスクがある", "ヤミカラス、ポリゴンを優先"),
    ("K-2", "ポケモン選択", "ヤミカラス系ラインがある", "ドンカラスを選ぶ"),
    ("K-3", "ポケモン選択", "ポリゴン系ラインがある", "ポリゴン2を選ぶ"),
    ("K-4", "ポケモン選択", "通常検索", "ヤミカラス、ドンカラス、ポリゴン、ポリゴン2の順"),
    ("K-D", "ドラパルト対面ポケモン選択", "相手盤面にドラパルトLineが1枚以上見えている", "フリーザー未設置ならフリーザーを最優先。設置後はヤミカラス2体、ポリゴンの順。フリーザー未設置で既存ポリゴンLineがある、または残りベンチ1枠ならポリゴン選択を止める。進化先はこのターン攻撃へ直結する時だけ選ぶ"),
    ("E-1", "エネルギー選択", "ロケット団エネルギーがある", "ロケット団エネルギーを選ぶ"),
    ("E-2", "エネルギー選択", "ロケット団エネルギーがなくイグニッションがある", "イグニッションを選ぶ"),
    ("C-1", "ファクトリー/その他選択", "ファクトリーが選択肢にある", "ファクトリーを選ぶ"),
    ("C-2", "ファクトリー/その他選択", "特別な選択肢に当てはまらない", "順位評価fallbackで選ぶ。必須選択でも先頭補完だけにしない"),
)


RULE_ONLY_DETAIL_PHASES_JA = RULE_ONLY_DETAIL_PHASES_JA + (
    ("1-A", "対フーディン進化ガード", "相手盤面にフーディンLineが見え、フリーザーがサイド落ち確定ではなく、残りサイドを取り切る準備済みアタッカー数も足りない", "ドンカラス/ポリゴン2への進化を止め、たねのロケット団ポケモンをレジストヴェールで守る"),
    ("2-A", "対フーディンたね展開", "対フーディン進化ガード中で、ベンチに空きがある", "フリーザーを最優先で置き、次に残りサイド数を意識してロケット団エネルギー付きヤミカラスを複数準備する"),
    ("3-A", "対フーディンフリーザー復旧", "対フーディン進化ガード中で、フリーザーが場におらずトラッシュにあり、ベンチへ戻せる", "フリーザーを直接置けるなら置く。夜のタンカがあれば使い、山札にしかなければラムダやレシーバー/ポケギア/ロトスティックから夜のタンカへつなぐ"),
    ("6-A", "対フーディン前ポリゴン脱出", "対フーディン進化ガード中で、前のポリゴンがいちゃもんロックを邪魔している", "逃げられるなら逃げる。逃げられない時は前ポリゴンへのイグニッションを例外許可し、山札保存域ではサカキも脱出候補にする"),
    ("7-A", "対フーディンいちゃもん固定", "対フーディン進化ガード中で、前のヤミカラスがいちゃもん系ワザを使える", "ロケットフェザー/Rコマンド/たぶらかすへ逃げず、いちゃもん系ワザだけを使う"),
    ("X-A", "対フーディン解除条件", "フリーザーがサイド落ち確定、または残りサイドを取り切れるだけの準備済みアタッカーがいる", "通常の進化・ロケットフェザー・Rコマンドのフローへ戻す"),
)

RULE_ONLY_OPTIONAL_DETAIL_PHASES_JA = RULE_ONLY_OPTIONAL_DETAIL_PHASES_JA + (
    ("P-A", "対フーディン前出し", "相手盤面にフーディンLineが見え、フリーザーが使用可能", "フリーザーは前に出さず、エネルギー付きヤミカラスを最優先して前に出す"),
    ("K-A", "対フーディンポケモン選択", "ポケパッド/夜のタンカ/ランス等のポケモン選択で、対フーディン進化ガード中", "フリーザーがトラッシュにある時は復旧を最優先。通常はフリーザー、必要数のヤミカラス、最低限のポリゴンを優先し、攻撃に直結しない進化先は選ばない"),
)


def _rule_option_id(observation, option):
    return _card_id(_card_from_option(observation, option))


def _rule_target_id(observation, option):
    return _card_id(_target_card_from_option(observation, option))


def _rule_option_is_play(option):
    return _option_type(option) in (7, "play")


def _rule_find_card_option(observation, options, card_ids, option_types=None):
    if isinstance(card_ids, int):
        card_ids = (card_ids,)
    for option_index, option in enumerate(options):
        if option_types is not None and _option_type(option) not in option_types:
            continue
        if _rule_option_id(observation, option) in card_ids:
            return option_index
    return None


def _rule_find_attack_option(options, attack_ids):
    if isinstance(attack_ids, int):
        attack_ids = (attack_ids,)
    for option_index, option in enumerate(options):
        if _option_type(option) in (13, "attack") and _attack_id(option) in attack_ids:
            return option_index
    return None


def _rule_find_retreat_option(options):
    for option_index, option in enumerate(options):
        if _option_type(option) in (12, "retreat", "switch"):
            return option_index
    return None


def _rule_supporter_access_after_spend(player, spent_supporters):
    hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    hand_supporters_after = max(0, hand_supporters - max(0, spent_supporters))
    deck_count = _deck_count(player)
    deck_supporters = _deck_card_count_for_policy(player, lambda card_id: card_id in ROCKET_SUPPORTERS)
    deck_search_outs = _deck_card_count_for_policy(player, lambda card_id: card_id in (TEAM_ROCKET_TRANSCEIVER, POKEGEAR, ROTO_STICK))
    hand_ids = _rule_hand_ids(player)
    discard_supporters = _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS)

    access = min(1.0, hand_supporters_after * 0.35)
    if deck_supporters > 0:
        if TEAM_ROCKET_TRANSCEIVER in hand_ids:
            access += 0.95
        if POKEGEAR in hand_ids:
            access += 0.60
        if ROTO_STICK in hand_ids:
            access += 0.45
    if discard_supporters > 0 and MIRACLE_HEADSET in hand_ids:
        access += 0.70

    if deck_count > 0:
        effective_outs = min(deck_count, deck_supporters + deck_search_outs)
        access += _energy_hit_probability(deck_count, effective_outs, min(2, deck_count)) * 0.75

    return min(1.0, access)


def _rule_attack_efficiency_score(observation, attack_id):
    current, _, player = _current_player(observation)
    target = _opponent_active_card(observation)
    remaining_hp = _remaining_hp(target)
    if target is None or remaining_hp <= 0:
        return -1_000_000

    prize_count = _prize_count_for_knockout(target)
    deck_count = _deck_count(player)
    if attack_id == ROCKET_FEATHER_ATTACK:
        hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)
        if hand_supporters <= 0:
            return -1_000_000
        damage_per_supporter = _rocket_feather_damage_per_supporter(target)
        if damage_per_supporter <= 0:
            return -1_000_000
        required_for_ko = max(1, (remaining_hp + damage_per_supporter - 1) // damage_per_supporter)
        can_ko = required_for_ko <= hand_supporters
        if can_ko:
            supporters_to_spend = required_for_ko
        else:
            supporters_to_spend = _non_ko_rocket_feather_split_fuel_count(
                player,
                remaining_hp,
                damage_per_supporter,
                hand_supporters,
            )
            if supporters_to_spend <= 0:
                return -1_000_000
        damage = damage_per_supporter * supporters_to_spend
        future_access = _rule_supporter_access_after_spend(player, supporters_to_spend)
        cost_penalty = int(supporters_to_spend * (13_500 - future_access * 7_500))
        if supporters_to_spend >= hand_supporters and future_access < 0.35:
            cost_penalty += 32_000
        if _non_ko_refill_supporter_to_preserve(player) is not None and supporters_to_spend >= hand_supporters and not can_ko:
            cost_penalty += 80_000

        score = min(damage, remaining_hp) * 720 - max(0, damage - remaining_hp) * 90 - cost_penalty
        if can_ko:
            score += 210_000 + prize_count * 95_000
        else:
            score -= 8_000
            if damage >= max(damage_per_supporter, remaining_hp // 2):
                score += 42_000
            if remaining_hp - damage <= damage_per_supporter * 2:
                score += 28_000
            if deck_count <= 8:
                score -= 24_000
        return score

    if attack_id in PORYGON2_R_COMMAND_ATTACKS:
        discard_supporters = _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS)
        damage = _damage_after_type_modifier(
            discard_supporters * _policy_rule_number("porygon2RCommandFallback", "damagePerTrashRocketSupporter", 20),
            target,
            "colorless",
        )
        if damage <= 0:
            return -1_000_000
        can_ko = damage >= remaining_hp
        score = min(damage, remaining_hp) * 690 - max(0, damage - remaining_hp) * 60
        if can_ko:
            score += 205_000 + prize_count * 95_000
            score += _porygon2_endgame_r_command_ko_bonus(player, can_ko)
        else:
            score -= 48_000
        if discard_supporters >= _porygon2_late_trash_threshold():
            score += 34_000
        return score

    return -1_000_000


def _rule_rocket_feather_ko_snapshot(observation):
    _, _, player = _current_player(observation)
    active = _top_card(_read(player, "active", []))
    if _card_id(active) != HONCHKROW:
        return None
    target = _opponent_active_card(observation)
    remaining_hp = _remaining_hp(target)
    if remaining_hp <= 0:
        return None
    damage_per_supporter = _rocket_feather_damage_per_supporter(target)
    if damage_per_supporter <= 0:
        return None
    hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    required_for_ko = max(1, (remaining_hp + damage_per_supporter - 1) // damage_per_supporter)
    discard_supporters = _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    return {
        "player": player,
        "target": target,
        "remaining_hp": remaining_hp,
        "damage_per_supporter": damage_per_supporter,
        "required_for_ko": required_for_ko,
        "hand_supporters": hand_supporters,
        "discard_supporters": discard_supporters,
        "prize_count": _prize_count_for_knockout(target),
    }


def _rule_rocket_feather_ko_reachable_from_hand(observation):
    snapshot = _rule_rocket_feather_ko_snapshot(observation)
    if snapshot is None or snapshot["hand_supporters"] <= 0:
        return False
    required_for_ko = snapshot["required_for_ko"]
    hand_supporters = snapshot["hand_supporters"]
    return required_for_ko <= hand_supporters


def _rule_draw_supporter_refuel_current_ko_probability(observation, supporter_id):
    snapshot = _rule_rocket_feather_ko_snapshot(observation)
    if snapshot is None:
        return 0.0
    player = snapshot["player"]
    required_for_ko = snapshot["required_for_ko"]
    hand_supporters = snapshot["hand_supporters"]
    deck_supporters = _deck_card_count_for_policy(player, lambda card_id: card_id in ROCKET_SUPPORTERS)
    deck_count = _deck_count(player)

    if supporter_id == ARIANA:
        retained_supporters = max(0, hand_supporters - 1)
        needed_from_draw = max(0, required_for_ko - retained_supporters)
        return _hit_at_least_probability(deck_count, deck_supporters, _ariana_draw_count_for_player(player), needed_from_draw)

    if supporter_id == ARCHER:
        hand_count = _hand_count(player)
        apollo_population = deck_count + max(0, hand_count - 1)
        apollo_supporters = deck_supporters + max(0, hand_supporters - 1)
        apollo_draw_count = min(5, apollo_population)
        return _hit_at_least_probability(apollo_population, apollo_supporters, apollo_draw_count, required_for_ko)

    return 0.0


def _rule_draw_supporter_can_refuel_current_ko(observation, supporter_id):
    if supporter_id not in (ARIANA, ARCHER):
        return False
    snapshot = _rule_rocket_feather_ko_snapshot(observation)
    if snapshot is None:
        return False
    probability = _rule_draw_supporter_refuel_current_ko_probability(observation, supporter_id)
    threshold = 0.72 if supporter_id == ARIANA else 0.78
    if snapshot["prize_count"] >= 2:
        threshold += 0.03
    return probability >= threshold


def _rule_rocket_feather_fuel_shortage(observation, min_prize=1):
    if _opponent_active_has_hop_dodge_protection(observation):
        return None
    snapshot = _rule_rocket_feather_ko_snapshot(observation)
    if snapshot is None or snapshot["prize_count"] < min_prize:
        return None
    shortage = snapshot["required_for_ko"] - snapshot["hand_supporters"]
    if shortage <= 0:
        return None
    snapshot = dict(snapshot)
    snapshot["shortage"] = shortage
    return snapshot


def _rule_rocket_feather_recoverable_supporters_with_miracle(observation):
    snapshot = _rule_rocket_feather_ko_snapshot(observation)
    if snapshot is None:
        return 0
    hand_ids = _rule_hand_ids(snapshot["player"])
    if MIRACLE_HEADSET not in hand_ids:
        return 0
    # Miracle Headset recovers up to two Pokemon/Trainer cards from discard.
    # For Rocket Feather lethals, any Rocket supporter recovered before the
    # attack is direct fuel.  Keep this deterministic and do not count future
    # draw/search guesses here.
    return min(2, snapshot["discard_supporters"])


def _rule_rocket_feather_ko_reachable_after_miracle(observation, min_prize=1):
    if _opponent_active_has_hop_dodge_protection(observation):
        return False
    snapshot = _rule_rocket_feather_ko_snapshot(observation)
    if snapshot is None:
        return False
    if snapshot["prize_count"] < min_prize:
        return False
    hand_supporters = snapshot["hand_supporters"]
    if hand_supporters >= snapshot["required_for_ko"]:
        return False
    recoverable = _rule_rocket_feather_recoverable_supporters_with_miracle(observation)
    return snapshot["required_for_ko"] <= hand_supporters + recoverable


def _rule_find_miracle_headset_high_prize_ko_option(observation, options):
    miracle = _rule_find_card_option(observation, options, MIRACLE_HEADSET, (7, "play"))
    if miracle is None:
        return None
    rocket_feather = _rule_find_attack_option(options, ROCKET_FEATHER_ATTACK)
    if rocket_feather is None:
        return None
    if _rule_rocket_feather_ko_reachable_after_miracle(observation, min_prize=2):
        return miracle
    return None


def _rule_find_miracle_headset_athena_rescue_option(observation, options):
    current, _, player = _current_player(observation)
    if _supporter_played_this_turn(current, player):
        return None
    miracle = _rule_find_card_option(observation, options, MIRACLE_HEADSET, (7, "play"))
    if miracle is None:
        return None
    hand_ids = _rule_hand_ids(player)
    if ARIANA in hand_ids:
        return None
    if _count_cards(player, ("discard",), lambda card_id: card_id == ARIANA) <= 0:
        return None
    if _rule_find_immediate_ko_attack_option(observation, options) is not None:
        return None
    if _rule_lance_opening_allowed_for_player(current, player) and PROTON in hand_ids:
        return None

    archer = _rule_find_supporter_option(observation, options, ARCHER)
    apollo_score = _rule_apollo_direct_play_score(observation) if archer is not None else None
    if apollo_score is not None and apollo_score >= 22_000:
        return None

    if (
        _needs_ariana_energy_dig(player)
        or _rocket_feather_fuel_needed(player)
        or _rule_board_collapse_reset_needed(observation)
    ):
        return miracle
    return None


def _rule_miracle_headset_rocket_feather_fuel_targets(observation, options, max_count, min_prize=2):
    if max_count <= 0 or _opponent_active_has_hop_dodge_protection(observation):
        return []
    snapshot = _rule_rocket_feather_ko_snapshot(observation)
    if snapshot is None or snapshot["prize_count"] < min_prize:
        return []
    shortage = snapshot["required_for_ko"] - snapshot["hand_supporters"]
    if shortage <= 0 or shortage > max_count:
        return []

    option_ids = [_rule_option_id(observation, option) for option in options]
    available_supporters = sum(1 for card_id in option_ids if card_id in ROCKET_SUPPORTERS)
    if available_supporters < shortage:
        return []

    selected = []
    for wanted_id in RULE_ONLY_MIRACLE_RECOVERY_ORDER:
        for option_index, option_id in enumerate(option_ids):
            if option_id == wanted_id and option_index not in selected:
                selected.append(option_index)
                if len(selected) >= shortage:
                    return selected
    return []


def _rule_miracle_headset_athena_rescue_targets(observation, options, max_count):
    if max_count <= 0:
        return []
    current, _, player = _current_player(observation)
    if ARIANA in _rule_hand_ids(player):
        return []
    option_ids = [_rule_option_id(observation, option) for option in options]
    selected = []
    for wanted_id in (ARIANA, ARCHER, PETREL):
        for option_index, option_id in enumerate(option_ids):
            if option_id == wanted_id and option_index not in selected:
                selected.append(option_index)
                if len(selected) >= max_count:
                    return selected
        if selected:
            return selected
    return []


def _rule_miracle_headset_recovery_targets(observation, options, min_count, max_count):
    selected = _rule_miracle_headset_rocket_feather_fuel_targets(observation, options, max_count)
    if selected:
        return selected
    selected = _rule_miracle_headset_athena_rescue_targets(observation, options, max_count)
    if selected:
        return selected
    if min_count <= 0:
        return []
    option_ids = [_rule_option_id(observation, option) for option in options]
    # Mandatory fallback: keep the best future supporter, but never let generic
    # receiver logic turn Miracle Headset into a Proton/Lance setup recovery.
    selected = _rule_select_option_ids(option_ids, (ARIANA, ARCHER, PETREL, GIOVANNI), max_count)
    if selected:
        return selected
    return []


def _rule_find_immediate_ko_attack_option(observation, options):
    if _alakazam_lock_strategy_active(observation):
        return None
    if _opponent_active_has_hop_dodge_protection(observation):
        return None

    _, _, player = _current_player(observation)
    if _needs_seed_out_bench_guard(player) and _has_basic_play_option(observation):
        return None

    target = _opponent_active_card(observation)
    remaining_hp = _remaining_hp(target)
    if target is None or remaining_hp <= 0:
        return None

    r_command = _rule_find_attack_option(options, PORYGON2_R_COMMAND_ATTACKS)
    r_command_can_ko = False
    if r_command is not None:
        discard_supporters = _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS)
        damage = _damage_after_type_modifier(
            discard_supporters * _policy_rule_number("porygon2RCommandFallback", "damagePerTrashRocketSupporter", 20),
            target,
            "colorless",
        )
        r_command_can_ko = damage >= remaining_hp
        if r_command_can_ko and _porygon2_endgame_r_command_ko_window(player):
            return r_command

    rocket_feather = _rule_find_attack_option(options, ROCKET_FEATHER_ATTACK)
    if rocket_feather is not None and _rule_rocket_feather_ko_reachable_from_hand(observation):
        return rocket_feather

    if r_command_can_ko:
        return r_command

    return None


def _rule_supporter_can_precede_current_ko_attack(observation, options, supporter_id):
    immediate_attack = _rule_find_immediate_ko_attack_option(observation, options)
    if immediate_attack is None:
        return True
    attack_id = _attack_id(options[immediate_attack])
    if attack_id in PORYGON2_R_COMMAND_ATTACKS:
        _, _, player = _current_player(observation)
        return not _porygon2_endgame_r_command_ko_window(player)
    if attack_id == ROCKET_FEATHER_ATTACK:
        snapshot = _rule_rocket_feather_ko_snapshot(observation)
        if snapshot is None:
            return False
        # Do not spend the exact Rocket supporter fuel needed for a current KO.
        return (
            snapshot["hand_supporters"] > snapshot["required_for_ko"]
            or _rule_draw_supporter_can_refuel_current_ko(observation, supporter_id)
        )
    return False


def _rule_choose_best_main_attack(observation, options):
    if _alakazam_lock_strategy_active(observation):
        return None
    candidates = []
    rocket_feather = _rule_find_attack_option(options, ROCKET_FEATHER_ATTACK)
    if rocket_feather is not None:
        candidates.append((_rule_attack_efficiency_score(observation, ROCKET_FEATHER_ATTACK), rocket_feather))
    r_command = _rule_find_attack_option(options, PORYGON2_R_COMMAND_ATTACKS)
    if r_command is not None:
        candidates.append((_rule_attack_efficiency_score(observation, next(iter(PORYGON2_R_COMMAND_ATTACKS))), r_command))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_index = candidates[0]
    return best_index if best_score > -500_000 else None


def _rule_find_alakazam_taunt_attack_option(observation, options):
    if not _alakazam_lock_strategy_active(observation):
        return None
    _, _, player = _current_player(observation)
    active = _top_card(_read(player, "active", []))
    if _card_id(active) != MURKROW:
        return None
    return _rule_find_attack_option(options, RULE_ONLY_MURKROW_KO_ATTACKS)


def _rule_has_meaningful_main_attack_option(observation, options):
    if _rule_find_immediate_ko_attack_option(observation, options) is not None:
        return True
    if _rule_choose_best_main_attack(observation, options) is not None:
        return True
    if _rule_find_murkrow_ko_attack_option(observation, options) is not None:
        return True
    return False


def _rule_find_tempo_attack_option(observation, options):
    alakazam_taunt = _rule_find_alakazam_taunt_attack_option(observation, options)
    if alakazam_taunt is not None:
        return alakazam_taunt

    _, _, player = _current_player(observation)
    hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    candidates = []
    for option_index, option in enumerate(options):
        if _option_type(option) not in (13, "attack"):
            continue
        attack_id = _attack_id(option)
        if attack_id == ROCKET_FEATHER_ATTACK:
            score = _rule_attack_efficiency_score(observation, ROCKET_FEATHER_ATTACK)
        elif attack_id in PORYGON2_R_COMMAND_ATTACKS:
            score = _rule_attack_efficiency_score(observation, next(iter(PORYGON2_R_COMMAND_ATTACKS)))
        elif attack_id in RULE_ONLY_MURKROW_KO_ATTACKS:
            score = 25_000
        elif attack_id == MURKROW_TEMPT_ATTACK:
            score = -4_000 if hand_supporters > 0 else 8_000
        else:
            score = 0
        if score <= -900_000:
            continue
        candidates.append((score, -option_index, option_index))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def _rule_find_murkrow_ko_attack_option(observation, options):
    _, _, player = _current_player(observation)
    active = _top_card(_read(player, "active", []))
    if _card_id(active) != MURKROW:
        return None
    if _opponent_active_has_hop_dodge_protection(observation):
        return None
    opponent_active = _opponent_active_card(observation)
    remaining_hp = _remaining_hp(opponent_active)
    if remaining_hp <= 0:
        return None
    if _murkrow_taunt_damage(opponent_active) < remaining_hp:
        return None
    return _rule_find_attack_option(options, RULE_ONLY_MURKROW_KO_ATTACKS)


def _rule_find_evolution_option(observation, options, evolution_id):
    for option_index, option in enumerate(options):
        if _option_type(option) in (9, "evolve") and _rule_option_id(observation, option) == evolution_id:
            if _alakazam_forbidden_evolution_option(observation, option):
                continue
            if _dragapult_forbidden_evolution_option(observation, option):
                continue
            return option_index
    return None


def _rule_find_basic_play_option(observation, options, preferred_ids):
    for preferred_id in preferred_ids:
        for option_index, option in enumerate(options):
            if _option_type(option) not in (7, "play"):
                continue
            if _rule_option_id(observation, option) != preferred_id:
                continue
            if _dragapult_forbidden_basic_play_option(observation, option):
                continue
            return option_index
    return None


def _rule_find_dragapult_basic_play_option(observation, options):
    if not _dragapult_matchup_active(observation):
        return None
    _, _, player = _current_player(observation)
    if _bench_top_count(player) >= _bench_limit(player):
        return None
    return _rule_find_basic_play_option(observation, options, _dragapult_basic_priority_ids(player))


def _rule_find_alakazam_basic_play_option(observation, options):
    if not _alakazam_resist_veil_plan_active(observation):
        return None
    _, _, player = _current_player(observation)
    if _bench_top_count(player) >= _bench_limit(player):
        return None
    return _rule_find_basic_play_option(observation, options, _alakazam_basic_priority_ids(observation))


def _rule_find_supporter_option(observation, options, supporter_id):
    return _rule_find_card_option(observation, options, supporter_id, (7, "play"))


def _rule_find_alakazam_articuno_recovery_option(observation, options):
    if not _alakazam_articuno_recovery_needed(observation):
        return None

    articuno = _rule_find_card_option(observation, options, ARTICUNO, (7, "play"))
    if articuno is not None:
        return articuno

    night_stretcher = _rule_find_card_option(observation, options, NIGHT_STRETCHER, (7, "play"))
    if night_stretcher is not None:
        return night_stretcher

    current, _, player = _current_player(observation)
    deck_ids = _deck_card_ids_for_policy(player)
    if NIGHT_STRETCHER in deck_ids and not _supporter_played_this_turn(current, player):
        petrel = _rule_find_supporter_option(observation, options, PETREL)
        if petrel is not None:
            return petrel
        if PETREL in deck_ids:
            for item_id in (TEAM_ROCKET_TRANSCEIVER, POKEGEAR, ROTO_STICK):
                item = _rule_find_card_option(observation, options, item_id, (7, "play"))
                if item is not None:
                    return item
    return None


def _rule_find_alakazam_porygon_escape_option(observation, options):
    if not _alakazam_porygon_active_escape_needed(observation):
        return None

    retreat = _rule_find_retreat_option(options)
    if retreat is not None:
        return retreat

    current, _, player = _current_player(observation)
    active = _top_card(_read(player, "active", []))
    if not _energy_attached_this_turn(current, player) and _attached_energy_cards(active) <= 0:
        for option_index, option in enumerate(options):
            if _option_type(option) not in (8, "attach"):
                continue
            if _rule_option_id(observation, option) != IGNITION_ENERGY:
                continue
            if _area_code(_read(option, "inPlayArea"), -1) != 4:
                continue
            target = _target_card_from_option(observation, option)
            if _card_id(target) == PORYGON and not _rule_forbidden_energy_attach_option(observation, option):
                return option_index

    preserve_threshold = _policy_rule_number("alakazamLockPlan", "deckPreserveThreshold", 20)
    if not _supporter_played_this_turn(current, player) and _deck_count(player) <= preserve_threshold:
        giovanni = _rule_find_supporter_option(observation, options, GIOVANNI)
        if giovanni is not None:
            return giovanni
    return None


def _rule_hand_ids(player):
    return [_card_id(card) for card in _iter_cards(_read(player, "hand", []))]


def _rule_field_top_ids(player):
    return [_card_id(card) for card in _field_top_cards(player)]


def _rule_stadium_ids(current):
    return [_card_id(card) for card in _iter_cards(_read(current, "stadium", []))]


def _rule_has_hand_supporter(player, supporter_id):
    return supporter_id in _rule_hand_ids(player)


def _rule_has_energy_in_hand(player):
    return any(card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY) for card_id in _rule_hand_ids(player))


def _rule_attacker_ready(card):
    identifier = _card_id(card)
    if identifier == HONCHKROW:
        return _attached_energy_cards(card) > 0
    if identifier == PORYGON2:
        return _attached_energy_cards(card) > 0
    return False


def _rule_target_has_energy(target):
    return _attached_energy_cards(target) > 0


def _rule_can_prepare_murkrow(player):
    hand_ids = _rule_hand_ids(player)
    deck_ids = _deck_card_ids_for_policy(player)
    return HONCHKROW in hand_ids or HONCHKROW in deck_ids


def _rule_setup_incomplete(player):
    field_ids = _field_card_ids(player)
    return (
        _zone_count(player, "bench") < 2
        or field_ids.count(MURKROW) + field_ids.count(HONCHKROW) < 2
    )


def _rule_petrel_poke_pad_bridge_needed(player):
    field_ids = _rule_field_top_ids(player)
    hand_ids = _rule_hand_ids(player)
    deck_ids = _deck_card_ids_for_policy(player)
    if POKE_PAD not in deck_ids:
        return False
    murkrow_lines = field_ids.count(MURKROW) + field_ids.count(HONCHKROW)
    if _zone_count(player, "bench") < 2 or murkrow_lines < 2:
        return True
    if MURKROW in field_ids and HONCHKROW not in hand_ids:
        return True
    if PORYGON in field_ids and PORYGON2 not in hand_ids and _porygon_development_allowed(player):
        return True
    return False


def _rule_petrel_energy_bridge_needed(player):
    hand_ids = _rule_hand_ids(player)
    if any(card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY) for card_id in hand_ids):
        return False
    return _needs_ariana_energy_dig(player) and any(
        card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY)
        for card_id in _deck_card_ids_for_policy(player)
    )


def _rule_petrel_bridge_needed(observation):
    current, _, player = _current_player(observation)
    if _alakazam_petrel_poke_pad_bridge_needed(observation):
        return True
    if FACTORY not in _rule_stadium_ids(current):
        return True
    if _rule_petrel_poke_pad_bridge_needed(player):
        return True
    if _rule_petrel_energy_bridge_needed(player):
        return True
    return False


def _rule_select_supporter_target(player):
    current = {}
    setup_incomplete = _rule_setup_incomplete(player)
    first_lance_pending = _proton_not_in_discard_or_trash(player)
    if first_lance_pending and setup_incomplete:
        return PROTON
    return ARIANA


def _rule_lance_opening_allowed_for_player(current, player):
    return _proton_opening_allowed(current, player, _rule_setup_incomplete(player))


def _rule_receiver_like_search_target(observation, options, max_count):
    current, _, player = _current_player(observation)
    setup_incomplete = _rule_setup_incomplete(player)
    first_lance_pending = _proton_not_in_discard_or_trash(player)
    wanted = []
    option_ids = [_rule_option_id(observation, option) for option in options]

    if _rule_lance_opening_allowed_for_player(current, player) and PROTON in option_ids:
        wanted.append(PROTON)
    else:
        if _alakazam_articuno_petrel_recovery_needed(observation):
            wanted.append(PETREL)
        if _sakaki_hop_dodge_escape_ko_score(observation, giovanni_from_hand=True) is not None:
            wanted.append(GIOVANNI)
        if _dragapult_articuno_search_needed(observation):
            wanted.append(PROTON)
        if _alakazam_petrel_poke_pad_bridge_needed(observation):
            wanted.append(PETREL)
        if _dragapult_petrel_poke_pad_attack_bridge_needed(observation):
            wanted.append(PETREL)
        energy_preference = _supporter_energy_dig_preference(current, player)
        if energy_preference is not None:
            preferred_name, _ = energy_preference
            if preferred_name == "apollo":
                wanted.extend((ARCHER, ARIANA))
            else:
                wanted.extend((ARIANA, ARCHER))
        wanted.extend(RULE_ONLY_SUPPORTER_AFTER_LANCE)

    selected = []
    for supporter_id in wanted:
        for option_index, option_id in enumerate(option_ids):
            if option_id == supporter_id and option_index not in selected:
                selected.append(option_index)
                if len(selected) >= max_count:
                    return selected
    if not selected:
        # Do not silently whiff a supporter search just because the perfect
        # supporter was not present.  After the opening Lance window, Lance is
        # intentionally last, but a single legal supporter is still better than
        # throwing away the search effect.
        for supporter_id in (ARIANA, ARCHER, PETREL, GIOVANNI, PROTON):
            for option_index, option_id in enumerate(option_ids):
                if option_id == supporter_id and option_index not in selected:
                    selected.append(option_index)
                    if len(selected) >= max_count:
                        return selected
    return selected


def _rule_pokemon_search_target(observation, options, max_count):
    current, _, player = _current_player(observation)
    field_ids = _rule_field_top_ids(player)
    hand_ids = _rule_hand_ids(player)
    active = _top_card(_read(player, "active", []))
    active_id = _card_id(active)
    option_ids = [_rule_option_id(observation, option) for option in options]
    wanted = []

    if _alakazam_resist_veil_plan_active(observation):
        wanted.extend(_alakazam_basic_priority_ids(observation))
        if not _alakazam_lock_strategy_active(observation):
            wanted.extend((HONCHKROW, PORYGON2))
        wanted.extend((ARTICUNO, MURKROW, PORYGON, HONCHKROW, PORYGON2))
        selected = []
        for wanted_id in wanted:
            for option_index, option_id in enumerate(option_ids):
                if option_id == wanted_id and option_index not in selected:
                    selected.append(option_index)
                    if len(selected) >= max_count:
                        return selected
        return selected

    if _dragapult_matchup_active(observation):
        wanted.extend(_dragapult_basic_priority_ids(player))
        if _dragapult_evolution_attack_ready(observation, HONCHKROW):
            wanted.append(HONCHKROW)
        if _dragapult_evolution_attack_ready(observation, PORYGON2):
            wanted.append(PORYGON2)
        wanted.extend((MURKROW, ARTICUNO, PORYGON, HONCHKROW, PORYGON2))
        selected = []
        for wanted_id in wanted:
            for option_index, option_id in enumerate(option_ids):
                if _dragapult_should_hold_porygon_for_articuno(observation, option_id):
                    continue
                if option_id == wanted_id and option_index not in selected:
                    selected.append(option_index)
                    if len(selected) >= max_count:
                        return selected
        return selected

    if _needs_seed_out_bench_guard(player):
        wanted.extend((MURKROW, PORYGON))
    if (
        active_id == MURKROW
        and HONCHKROW in option_ids
        and not bool(_read(active, "appearThisTurn", False))
        and not _needs_seed_out_bench_guard(player)
    ):
        wanted.append(HONCHKROW)
    if _rule_setup_incomplete(player):
        wanted.extend((MURKROW, PORYGON))
    if MURKROW in field_ids or MURKROW in hand_ids or active_id == MURKROW:
        wanted.append(HONCHKROW)
    if PORYGON in field_ids or PORYGON in hand_ids or active_id == PORYGON:
        wanted.append(PORYGON2)
    wanted.extend((MURKROW, HONCHKROW, PORYGON, PORYGON2))

    selected = []
    for wanted_id in wanted:
        for option_index, option_id in enumerate(option_ids):
            if option_id == wanted_id and option_index not in selected:
                selected.append(option_index)
                if len(selected) >= max_count:
                    return selected
    return selected


def _rule_energy_search_target(observation, options, max_count):
    selected = []
    option_ids = [_rule_option_id(observation, option) for option in options]
    for wanted_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY):
        for option_index, option_id in enumerate(option_ids):
            if option_id == wanted_id and option_index not in selected:
                selected.append(option_index)
                if len(selected) >= max_count:
                    return selected
    return selected


def _rule_effect_id(observation):
    select = _read(observation, "select", {})
    for key in ("effect", "contextCard", "source", "sourceCard"):
        identifier = _card_id(_read(select, key))
        if identifier is not None:
            return identifier
    return None


def _rule_select_option_ids(option_ids, wanted_ids, max_count):
    selected = []
    for wanted_id in wanted_ids:
        for option_index, option_id in enumerate(option_ids):
            if option_id == wanted_id and option_index not in selected:
                selected.append(option_index)
                if len(selected) >= max_count:
                    return selected
    return selected


def _rule_active_rocket_feather_ready(player):
    active = _top_card(_read(player, "active", []))
    return _card_id(active) == HONCHKROW and _attached_energy_cards(active) > 0


def _rule_tool_or_support_target(observation, options, max_count):
    option_ids = [_rule_option_id(observation, option) for option in options]
    current, _, player = _current_player(observation)
    field_ids = _rule_field_top_ids(player)
    hand_ids = _rule_hand_ids(player)
    stadium_ids = _rule_stadium_ids(current)
    wanted = []

    effect_id = _rule_effect_id(observation)
    if effect_id == FACTORY and _rule_active_rocket_feather_ready(player) and any(card_id in ROCKET_SUPPORTERS for card_id in option_ids):
        selected = _rule_receiver_like_search_target(observation, options, max_count)
        if selected:
            return selected

    if _alakazam_articuno_petrel_recovery_needed(observation) and PETREL in option_ids:
        return _rule_select_option_ids(option_ids, (PETREL,), max_count)

    if _alakazam_resist_veil_plan_active(observation):
        wanted.extend(_alakazam_basic_priority_ids(observation))

    if FACTORY not in stadium_ids:
        wanted.append(FACTORY)
    if not any(card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY) for card_id in hand_ids):
        wanted.extend((TEAM_ROCKET_ENERGY, IGNITION_ENERGY))
    if _rule_setup_incomplete(player):
        wanted.extend((MURKROW, PORYGON, HONCHKROW, PORYGON2))
    if MURKROW in field_ids or MURKROW in hand_ids:
        wanted.append(HONCHKROW)
    if PORYGON in field_ids or PORYGON in hand_ids:
        wanted.append(PORYGON2)
    wanted.extend((TEAM_ROCKET_TRANSCEIVER, POKEGEAR, POKE_PAD, ROTO_STICK, NIGHT_STRETCHER, ARIANA, ARCHER, PETREL))
    if _rule_lance_opening_allowed_for_player(current, player):
        wanted.append(PROTON)
    selected = _rule_select_option_ids(option_ids, wanted, max_count)
    if selected:
        return selected
    if any(card_id in ROCKET_SUPPORTERS for card_id in option_ids):
        selected = _rule_receiver_like_search_target(observation, options, max_count)
        if selected:
            return selected
    # Generic search effects such as Factory/Roto-Stick can reveal only a
    # narrow set.  Keep the choice meaningful instead of returning [] and
    # letting minCount fallback select an arbitrary first option.
    fallback_order = (
        ARIANA,
        ARCHER,
        PETREL,
        GIOVANNI,
        PROTON,
        TEAM_ROCKET_ENERGY,
        IGNITION_ENERGY,
        HONCHKROW,
        MURKROW,
        PORYGON2,
        PORYGON,
        TEAM_ROCKET_TRANSCEIVER,
        POKEGEAR,
        POKE_PAD,
        ROTO_STICK,
        NIGHT_STRETCHER,
        MIRACLE_HEADSET,
        FACTORY,
    )
    return _rule_select_option_ids(option_ids, fallback_order, max_count)


def _rule_ranked_optional_fallback(observation, options, min_count, max_count, preferred_ids=()):
    option_ids = [_rule_option_id(observation, option) for option in options]
    selected = _rule_select_option_ids(option_ids, preferred_ids, max_count) if preferred_ids else []
    if selected:
        return selected
    try:
        selected = _choose_optional_cards(observation, options, max_count)
        if selected:
            return selected[:max_count]
    except Exception:
        selected = []
    if len(options) == 1 and (min_count > 0 or option_ids[0] is not None):
        return [0]
    if min_count > 0:
        ranked = []
        for option_index, option in enumerate(options):
            try:
                ranked.append((_rank_optional_card(observation, option), option_index))
            except Exception:
                ranked.append((0, option_index))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [option_index for _, option_index in ranked[: min(min_count, max_count, len(options))]]
    return []


def _rule_choose_effect_targets(observation, options, min_count, max_count):
    if max_count <= 0:
        return []

    effect_id = _rule_effect_id(observation)
    option_ids = [_rule_option_id(observation, option) for option in options]
    current, _, player = _current_player(observation)
    hand_ids = _rule_hand_ids(player)
    stadium_ids = _rule_stadium_ids(current)

    if effect_id in (PROTON, POKE_PAD, NIGHT_STRETCHER):
        selected = _rule_pokemon_search_target(observation, options, max_count)
        if selected:
            return selected
        fallback_order = (
            _alakazam_basic_priority_ids(observation)
            if _alakazam_lock_strategy_active(observation)
            else (MURKROW, ARTICUNO, HONCHKROW, PORYGON, PORYGON2)
        )
        return _rule_ranked_optional_fallback(
            observation,
            options,
            min_count,
            max_count,
            fallback_order,
        )

    if effect_id == MIRACLE_HEADSET:
        selected = _rule_miracle_headset_recovery_targets(observation, options, min_count, max_count)
        if selected:
            return selected
        return _rule_ranked_optional_fallback(
            observation,
            options,
            min_count,
            max_count,
            (ARIANA, ARCHER, PETREL, GIOVANNI),
        )

    if effect_id in (TEAM_ROCKET_TRANSCEIVER, POKEGEAR):
        if _alakazam_articuno_petrel_recovery_needed(observation):
            selected = _rule_select_option_ids(option_ids, (PETREL,), max_count)
            if selected:
                return selected
        selected = _rule_receiver_like_search_target(observation, options, max_count)
        if selected:
            return selected
        return _rule_ranked_optional_fallback(
            observation,
            options,
            min_count,
            max_count,
            (ARIANA, ARCHER, PETREL, GIOVANNI, PROTON),
        )

    if effect_id == PETREL:
        wanted = []
        if _alakazam_articuno_recovery_needed(observation):
            wanted.append(NIGHT_STRETCHER)
        if _alakazam_petrel_poke_pad_bridge_needed(observation):
            wanted.append(POKE_PAD)
        if _rule_petrel_poke_pad_bridge_needed(player):
            wanted.append(POKE_PAD)
        if FACTORY not in stadium_ids:
            wanted.append(FACTORY)
        if _rule_petrel_energy_bridge_needed(player):
            wanted.extend((TEAM_ROCKET_ENERGY, IGNITION_ENERGY))
        wanted.extend((POKE_PAD, TEAM_ROCKET_TRANSCEIVER, POKEGEAR, ROTO_STICK, MIRACLE_HEADSET))
        selected = _rule_select_option_ids(option_ids, wanted, max_count)
        if selected:
            return selected
        selected = _rule_receiver_like_search_target(observation, options, max_count)
        if selected:
            return selected
        selected = _rule_pokemon_search_target(observation, options, max_count)
        if selected:
            return selected
        return _rule_ranked_optional_fallback(
            observation,
            options,
            min_count,
            max_count,
            (
                FACTORY,
                TEAM_ROCKET_ENERGY,
                IGNITION_ENERGY,
                POKE_PAD,
                TEAM_ROCKET_TRANSCEIVER,
                POKEGEAR,
                ROTO_STICK,
                MIRACLE_HEADSET,
                ARIANA,
                ARCHER,
                PETREL,
                GIOVANNI,
                PROTON,
                MURKROW,
                ARTICUNO,
                HONCHKROW,
                PORYGON,
                PORYGON2,
            ),
        )

    if effect_id in (ROTO_STICK, FACTORY):
        if effect_id == ROTO_STICK and _alakazam_articuno_petrel_recovery_needed(observation):
            selected = _rule_select_option_ids(option_ids, (PETREL,), max_count)
            if selected:
                return selected
        selected = _rule_tool_or_support_target(observation, options, max_count)
        if selected:
            return selected
        return _rule_ranked_optional_fallback(observation, options, min_count, max_count)

    return []


def _rule_choose_optional_cards(observation, options, min_count, max_count):
    if max_count <= 0:
        return []

    selected = _rule_choose_effect_targets(observation, options, min_count, max_count)
    if selected:
        return selected

    option_ids = [_rule_option_id(observation, option) for option in options]
    if any(card_id in ROCKET_SUPPORTERS for card_id in option_ids):
        selected = _rule_receiver_like_search_target(observation, options, max_count)
        if selected:
            return selected

    if any(card_id in (MURKROW, ARTICUNO, HONCHKROW, PORYGON, PORYGON2) for card_id in option_ids):
        selected = _rule_pokemon_search_target(observation, options, max_count)
        if selected:
            return selected

    if any(card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY) for card_id in option_ids):
        selected = _rule_energy_search_target(observation, options, max_count)
        if selected:
            return selected

    if FACTORY in option_ids:
        return [option_ids.index(FACTORY)]

    if min_count == 0:
        return []
    return _rule_ranked_optional_fallback(observation, options, min_count, max_count)


def _rule_choose_rocket_feather_costs(observation, options, max_count):
    select = _read(observation, "select", {})
    effect = _read(select, "effect")
    context = _select_context(select)
    # context 8 is also used by non-Honchkrow effects in the CABT engine.
    # Treat it as Rocket Feather cost only when the source effect is actually
    # Honchkrow; otherwise Porygon/other mandatory choices get corrupted.
    if _card_id(effect) != HONCHKROW:
        return None

    current = _read(observation, "current", {})
    your_index = _safe_int(_read(current, "yourIndex"), 0)
    player = _player_state(current, your_index)
    opponent = _player_state(current, 1 - your_index)
    opponent_active = _top_card(_read(opponent, "active", []))
    min_count, _ = _selection_bounds(select)
    remaining_hp = _remaining_hp(opponent_active)
    damage_per_supporter = _rocket_feather_damage_per_supporter(opponent_active)
    required_for_ko = max(1, (remaining_hp + damage_per_supporter - 1) // damage_per_supporter) if remaining_hp > 0 else 1
    can_ko = remaining_hp > 0 and required_for_ko <= max_count

    option_ids = [_rule_option_id(observation, option) for option in options]
    ordered = []
    for wanted_id in RULE_ONLY_FUEL_DISCARD_ORDER:
        for option_index, option_id in enumerate(option_ids):
            if option_id == wanted_id and option_index not in ordered:
                ordered.append(option_index)

    if not ordered:
        return []

    required = min(max_count, required_for_ko if can_ko else min_count)
    ordered_candidates = [(0, option_index, option_ids[option_index]) for option_index in ordered]
    if can_ko:
        ordered_candidates = _rocket_feather_cost_candidates_preserving_refill(ordered_candidates, required)
        ordered = [option_index for _, option_index, _ in ordered_candidates]
    else:
        ordered_candidates = _non_ko_cost_candidates_preserving_ariana(ordered_candidates)
        ordered = [option_index for _, option_index, _ in ordered_candidates]
        required = min(
            max_count,
            max(
                min_count,
                _non_ko_rocket_feather_split_fuel_count(
                    player,
                    remaining_hp,
                    damage_per_supporter,
                    max_count,
                ),
            ),
        )
        if required <= 0:
            return []
    return ordered[:required]


def _rule_choose_sakaki_preserving_promotion(observation, promotion_candidates, hand_ids):
    if GIOVANNI not in hand_ids or len(promotion_candidates) < 2:
        return None

    _, _, player = _current_player(observation)
    min_prize = _policy_rule_number("sakakiRequiresKo", "minPrizeScore", 2)
    targets = []
    for target in _opponent_bench_cards(observation):
        remaining_hp = _remaining_hp(target)
        prize_count = _prize_count_for_knockout(target)
        if remaining_hp > 0 and prize_count >= min_prize:
            targets.append((target, remaining_hp, prize_count))
    if not targets:
        return None

    best = None
    for promote_index, promote_option, promote_id in promotion_candidates:
        promote_card = _card_from_option(observation, promote_option)
        if promote_id not in ROCKET_FIELD_POKEMON:
            continue
        if promote_id == ARTICUNO:
            continue

        best_line_score = None
        for attacker_index, attacker_option, attacker_id in promotion_candidates:
            if attacker_index == promote_index or attacker_id not in (HONCHKROW, MURKROW, PORYGON2):
                continue
            attacker = _card_from_option(observation, attacker_option)
            for target, remaining_hp, prize_count in targets:
                damage = _bench_candidate_damage_after_sakaki(observation, attacker, target, giovanni_from_hand=True)
                if damage < remaining_hp:
                    continue
                line_score = 500_000 + prize_count * 120_000 + min(damage, 360) * 20
                if attacker_id == HONCHKROW:
                    line_score += 18_000
                elif attacker_id == PORYGON2:
                    line_score += 10_000
                elif attacker_id == MURKROW:
                    line_score += 6_000
                if best_line_score is None or line_score > best_line_score:
                    best_line_score = line_score

        if best_line_score is None:
            continue

        pivot_bonus = {
            PORYGON: 18_000,
            MURKROW: 10_000,
            PORYGON2: -6_000,
            HONCHKROW: -14_000,
        }.get(promote_id, 0)
        if _is_sakaki_ready_bench_attacker(promote_card):
            pivot_bonus -= 90_000

        score = best_line_score + pivot_bonus
        if best is None or score > best[0]:
            best = (score, promote_index)

    return best[1] if best is not None else None


def _rule_choose_promotion(observation, options, max_count):
    if max_count != 1:
        return None
    select = _read(observation, "select", {})
    if _select_context(select) not in (4, "promote active pokemon", "context4"):
        return None
    current, _, player = _current_player(observation)
    hand_ids = _rule_hand_ids(player)
    deck_ids = _deck_card_ids_for_policy(player)
    promotion_candidates = []
    for option_index, option in enumerate(options):
        option_id = _rule_option_id(observation, option)
        if option_id is not None:
            promotion_candidates.append((option_index, option, option_id))

    if _alakazam_resist_veil_plan_active(observation) and promotion_candidates:
        scored = [
            (_alakazam_promotion_score(observation, option), option_index)
            for option_index, option, _ in promotion_candidates
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [scored[0][1]]

    if _dragapult_matchup_active(observation) and promotion_candidates:
        scored = [
            (_dragapult_promotion_score(observation, option), option_index)
            for option_index, option, _ in promotion_candidates
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [scored[0][1]]

    sakaki_promotion = _rule_choose_sakaki_preserving_promotion(observation, promotion_candidates, hand_ids)
    if sakaki_promotion is not None:
        return [sakaki_promotion]

    for wanted_id in (HONCHKROW, PORYGON2):
        for option_index, option, option_id in promotion_candidates:
            if option_id == wanted_id and _rule_attacker_ready(_card_from_option(observation, option)):
                return [option_index]

    if HONCHKROW in hand_ids:
        for option_index, option, option_id in promotion_candidates:
            candidate = _card_from_option(observation, option)
            if option_id == MURKROW and _attached_energy_cards(candidate) > 0:
                return [option_index]

    preferred = [HONCHKROW]
    preferred.extend((PORYGON2, MURKROW, PORYGON))
    for wanted_id in preferred:
        for option_index, _, option_id in promotion_candidates:
            if option_id == wanted_id:
                return [option_index]
    return [0] if options else None


def _rule_choose_optional_multi_select(observation, options, min_count, max_count):
    mulligan_bonus_draw = _choose_mulligan_bonus_draw(observation, options, max_count)
    if mulligan_bonus_draw is not None:
        return mulligan_bonus_draw

    promotion = _rule_choose_promotion(observation, options, max_count)
    if promotion is not None:
        return promotion

    taunt_lock = _choose_taunt_move_lock_option(observation, options, max_count)
    if taunt_lock is not None:
        return taunt_lock

    rocket_feather_costs = _rule_choose_rocket_feather_costs(observation, options, max_count)
    if rocket_feather_costs is not None:
        return rocket_feather_costs

    selected = _rule_choose_optional_cards(observation, options, min_count, max_count)
    if selected or min_count == 0:
        return selected
    return list(range(min(min_count, len(options))))


def _rule_active_porygon_line_r_command_can_ko(observation):
    _, _, player = _current_player(observation)
    active = _top_card(_read(player, "active", []))
    active_id = _card_id(active)
    if active_id == PORYGON:
        return False
    if active_id != PORYGON2:
        return False
    opponent_active = _opponent_active_card(observation)
    remaining_hp = _remaining_hp(opponent_active)
    if remaining_hp <= 0:
        return False
    discard_supporters = _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    damage = discard_supporters * _policy_rule_number("porygon2RCommandFallback", "damagePerTrashRocketSupporter", 20)
    damage = _damage_after_type_modifier(damage, opponent_active, "colorless")
    return damage >= remaining_hp


def _rule_r_command_pressure_is_worth_ignition(observation):
    _, _, player = _current_player(observation)
    opponent_active = _opponent_active_card(observation)
    remaining_hp = _remaining_hp(opponent_active)
    if remaining_hp <= 0:
        return False
    discard_supporters = _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    damage = discard_supporters * _policy_rule_number("porygon2RCommandFallback", "damagePerTrashRocketSupporter", 20)
    damage = _damage_after_type_modifier(damage, opponent_active, "colorless")
    if damage >= remaining_hp:
        return True
    if _prize_count_for_knockout(opponent_active) <= 1:
        return False
    return damage * 2 >= remaining_hp


def _rule_active_porygon_line_r_command_is_worth_ignition(observation):
    _, _, player = _current_player(observation)
    active = _top_card(_read(player, "active", []))
    active_id = _card_id(active)
    if active_id == PORYGON:
        return PORYGON2 in _rule_hand_ids(player) and _rule_r_command_pressure_is_worth_ignition(observation)
    if active_id == PORYGON2:
        return _rule_r_command_pressure_is_worth_ignition(observation)
    return False


def _rule_forbidden_energy_attach_option(observation, option):
    if _option_type(option) not in (8, "attach"):
        return False

    energy_id = _rule_option_id(observation, option)
    if energy_id not in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY):
        return False

    current, _, player = _current_player(observation)
    hand_ids = _rule_hand_ids(player)
    target = _target_card_from_option(observation, option)
    target_id = _card_id(target)
    if target_id is None:
        return True
    area = _area_code(_read(option, "inPlayArea"), -1)
    target_is_active = area == 4
    if target_id == ARTICUNO:
        return True
    if _alakazam_lock_strategy_active(observation) and target_id != MURKROW:
        if not (
            energy_id == IGNITION_ENERGY
            and target_id == PORYGON
            and target_is_active
            and _alakazam_porygon_active_escape_needed(observation)
        ):
            return True

    # Donkrow deck attacks are paid by one attached energy.  Any additional
    # manual energy on the same Pokemon is treated as waste and must not be
    # selected by the rule-only engine.
    if _attached_energy_cards(target) > 0:
        return True

    if energy_id == TEAM_ROCKET_ENERGY:
        return target_id not in (MURKROW, HONCHKROW)

    if energy_id == IGNITION_ENERGY:
        if target_id == MURKROW:
            return True
        if target_id == HONCHKROW:
            return (
                not target_is_active
                or TEAM_ROCKET_ENERGY in hand_ids
                or _opponent_active_has_hop_dodge_protection(observation)
                or not (
                    _rule_rocket_feather_ko_reachable_from_hand(observation)
                    or _rule_rocket_feather_ko_reachable_after_miracle(observation, min_prize=2)
                    or _active_high_prize_pressure_without_sakaki(observation)
                )
            )
        if target_id == PORYGON:
            if target_is_active and _alakazam_porygon_active_escape_needed(observation):
                return False
            return not target_is_active or not _rule_active_porygon_line_r_command_is_worth_ignition(observation)
        if target_id == PORYGON2:
            return not target_is_active or not _rule_active_porygon_line_r_command_is_worth_ignition(observation)
        return True

    return False


def _rule_find_opening_team_rocket_attach_option(observation, options):
    current, your_index, player = _current_player(observation)
    order = _opening_turn_order(current, your_index)
    if order is None:
        return None

    hand_ids = _rule_hand_ids(player)
    if TEAM_ROCKET_ENERGY not in hand_ids:
        return None

    scored = []
    for option_index, option in enumerate(options):
        if _option_type(option) not in (8, "attach"):
            continue
        if _rule_option_id(observation, option) != TEAM_ROCKET_ENERGY:
            continue
        if _rule_forbidden_energy_attach_option(observation, option):
            continue

        target = _target_card_from_option(observation, option)
        target_id = _card_id(target)
        target_energy_cards = _attached_energy_cards(target)
        target_has_rocket_energy = TEAM_ROCKET_ENERGY in _attached_energy_card_ids(target)
        target_is_active = _area_code(_read(option, "inPlayArea"), -1) == 4
        target_is_bench = _area_code(_read(option, "inPlayArea"), -1) == 5
        score = _opening_turn_team_rocket_energy_score(
            current,
            player,
            your_index,
            hand_ids,
            target_id,
            target_is_active,
            target_is_bench,
            target_energy_cards,
            target_has_rocket_energy,
        )
        if score is not None and score > 0:
            scored.append((score, option_index))

    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def _rule_find_attach_option(observation, options):
    current, your_index, player = _current_player(observation)
    hand_ids = _rule_hand_ids(player)
    field_top_cards = _field_top_cards(player)
    active = _top_card(_read(player, "active", []))
    active_id = _card_id(active)
    has_attack = _has_any_main_attack_option(observation)

    def attach_candidates(energy_id, target_ids, require_no_energy=True, active_only=False, bench_only=False):
        result = []
        for option_index, option in enumerate(options):
            if _option_type(option) not in (8, "attach"):
                continue
            if _rule_option_id(observation, option) != energy_id:
                continue
            if _rule_forbidden_energy_attach_option(observation, option):
                continue
            target = _target_card_from_option(observation, option)
            target_id = _card_id(target)
            if target_id not in target_ids:
                continue
            if require_no_energy and _rule_target_has_energy(target):
                continue
            area = _area_code(_read(option, "inPlayArea"), -1)
            if active_only and area != 4:
                continue
            if bench_only and area != 5:
                continue
            result.append(option_index)
        return result

    if _opening_turn_order(current, your_index) is not None and TEAM_ROCKET_ENERGY in hand_ids:
        opening_attach = _rule_find_opening_team_rocket_attach_option(observation, options)
        if opening_attach is not None:
            return opening_attach

    if active_id in (MURKROW, HONCHKROW) and not _rule_target_has_energy(active):
        active_attach = attach_candidates(TEAM_ROCKET_ENERGY, (active_id,), active_only=True)
        if active_attach:
            return active_attach[0]
        if (
            active_id == HONCHKROW
            and TEAM_ROCKET_ENERGY not in hand_ids
            and not _opponent_active_has_hop_dodge_protection(observation)
            and (
                _rule_rocket_feather_ko_reachable_from_hand(observation)
                or _rule_rocket_feather_ko_reachable_after_miracle(observation, min_prize=2)
                or _active_high_prize_pressure_without_sakaki(observation)
            )
        ):
            active_ignition = attach_candidates(IGNITION_ENERGY, (HONCHKROW,), active_only=True)
            if active_ignition:
                return active_ignition[0]

    if active_id == PORYGON2 and not _rule_target_has_energy(active):
        active_attach = attach_candidates(IGNITION_ENERGY, (PORYGON2,), active_only=True)
        if active_attach and _rule_active_porygon_line_r_command_is_worth_ignition(observation):
            return active_attach[0]

    if active_id == PORYGON and not _rule_target_has_energy(active) and _alakazam_porygon_active_escape_needed(observation):
        active_escape = attach_candidates(IGNITION_ENERGY, (PORYGON,), active_only=True)
        if active_escape:
            return active_escape[0]

    if has_attack:
        for target_ids in ((HONCHKROW,), (MURKROW,)):
            bench_attach = attach_candidates(TEAM_ROCKET_ENERGY, target_ids, bench_only=True)
            if bench_attach:
                return bench_attach[0]
        return None

    candidate = attach_candidates(TEAM_ROCKET_ENERGY, (HONCHKROW,))
    if candidate:
        return candidate[0]
    candidate = attach_candidates(TEAM_ROCKET_ENERGY, (MURKROW,))
    if candidate:
        return candidate[0]
    return None


def _rule_find_pre_support_attach_option(observation, options):
    current, _, player = _current_player(observation)
    if _supporter_played_this_turn(current, player) or _energy_attached_this_turn(current, player):
        return None

    hand_ids = _rule_hand_ids(player)
    your_index = _safe_int(_read(current, "yourIndex"), 0)
    if _opening_turn_order(current, your_index) is not None and TEAM_ROCKET_ENERGY in hand_ids:
        opening_attach = _rule_find_opening_team_rocket_attach_option(observation, options)
        if opening_attach is not None:
            return opening_attach

    active = _top_card(_read(player, "active", []))
    active_id = _card_id(active)

    def attach_candidates(energy_id, target_ids, require_no_energy=True, active_only=False, bench_only=False):
        result = []
        for option_index, option in enumerate(options):
            if _option_type(option) not in (8, "attach"):
                continue
            if _rule_option_id(observation, option) != energy_id:
                continue
            if _rule_forbidden_energy_attach_option(observation, option):
                continue
            target = _target_card_from_option(observation, option)
            target_id = _card_id(target)
            if target_id not in target_ids:
                continue
            if require_no_energy and _rule_target_has_energy(target):
                continue
            area = _area_code(_read(option, "inPlayArea"), -1)
            if active_only and area != 4:
                continue
            if bench_only and area != 5:
                continue
            result.append(option_index)
        return result

    if active_id in (MURKROW, HONCHKROW) and not _rule_target_has_energy(active):
        active_rocket = attach_candidates(TEAM_ROCKET_ENERGY, (active_id,), active_only=True)
        if active_rocket:
            return active_rocket[0]
        if (
            active_id == HONCHKROW
            and TEAM_ROCKET_ENERGY not in hand_ids
            and not _opponent_active_has_hop_dodge_protection(observation)
            and (
                _rule_rocket_feather_ko_reachable_from_hand(observation)
                or _rule_rocket_feather_ko_reachable_after_miracle(observation, min_prize=2)
                or _active_high_prize_pressure_without_sakaki(observation)
            )
        ):
            active_ignition = attach_candidates(IGNITION_ENERGY, (HONCHKROW,), active_only=True)
            if active_ignition:
                return active_ignition[0]

    if active_id == PORYGON2 and not _rule_target_has_energy(active) and _rule_active_porygon_line_r_command_is_worth_ignition(observation):
        active_ignition = attach_candidates(IGNITION_ENERGY, (active_id,), active_only=True)
        if active_ignition:
            return active_ignition[0]

    if active_id == PORYGON and not _rule_target_has_energy(active) and _rule_active_porygon_line_r_command_is_worth_ignition(observation):
        active_ignition = attach_candidates(IGNITION_ENERGY, (PORYGON,), active_only=True)
        if active_ignition:
            return active_ignition[0]

    if active_id == PORYGON and not _rule_target_has_energy(active) and _alakazam_porygon_active_escape_needed(observation):
        active_escape = attach_candidates(IGNITION_ENERGY, (PORYGON,), active_only=True)
        if active_escape:
            return active_escape[0]

    for target_ids in ((HONCHKROW,), (MURKROW,)):
        bench_rocket = attach_candidates(TEAM_ROCKET_ENERGY, target_ids, bench_only=True)
        if bench_rocket:
            return bench_rocket[0]

    return None


def _rule_factory_can_be_used_after_supporter(observation, options):
    current, _, player = _current_player(observation)
    if _supporter_played_this_turn(current, player):
        return False
    if FACTORY in _rule_stadium_ids(current):
        return True
    if _stadium_played_this_turn(current, player):
        return False
    return _rule_find_card_option(observation, options, FACTORY, (7, "play")) is not None


def _rule_find_factory_enabler_supporter(observation, options):
    if not _rule_factory_can_be_used_after_supporter(observation, options):
        return None

    petrel = _rule_find_supporter_option(observation, options, PETREL)
    if petrel is not None and _rule_supporter_can_precede_current_ko_attack(observation, options, PETREL):
        return petrel

    # Last fallback to unlock Factory's draw when the turn otherwise stalls.
    # Avoid Giovanni because an unneeded switch can erase an attack line.
    proton = _rule_find_supporter_option(observation, options, PROTON)
    if proton is not None and _rule_supporter_can_precede_current_ko_attack(observation, options, PROTON):
        return proton
    return None


def _rule_find_supporter_to_play(observation, options):
    current, _, player = _current_player(observation)
    if _supporter_played_this_turn(current, player):
        return None

    setup_incomplete = _rule_setup_incomplete(player)
    if _proton_opening_allowed(current, player, setup_incomplete):
        proton = _rule_find_supporter_option(observation, options, PROTON)
        if proton is not None and _rule_supporter_can_precede_current_ko_attack(observation, options, PROTON):
            return proton

    giovanni = _rule_find_supporter_option(observation, options, GIOVANNI)
    if giovanni is not None and _sakaki_hop_dodge_escape_ko_score(observation, giovanni_from_hand=True) is not None:
        return giovanni
    if giovanni is not None and _sakaki_prize_race_ko_score(observation, giovanni_from_hand=True) is not None:
        return giovanni
    if _dragapult_articuno_search_needed(observation):
        proton = _rule_find_supporter_option(observation, options, PROTON)
        if proton is not None and _rule_supporter_can_precede_current_ko_attack(observation, options, PROTON):
            return proton
    if _rule_low_deck_draw_search_blocked(observation, options):
        return None

    petrel = _rule_find_supporter_option(observation, options, PETREL)
    if (
        petrel is not None
        and _alakazam_petrel_poke_pad_bridge_needed(observation)
        and _rule_supporter_can_precede_current_ko_attack(observation, options, PETREL)
    ):
        return petrel
    if (
        petrel is not None
        and _dragapult_petrel_poke_pad_attack_bridge_needed(observation)
        and _rule_supporter_can_precede_current_ko_attack(observation, options, PETREL)
    ):
        return petrel

    energy_preference = _supporter_energy_dig_preference(current, player)
    if energy_preference is not None:
        preferred_name, _ = energy_preference
        if preferred_name == "apollo":
            archer = _rule_find_supporter_option(observation, options, ARCHER)
            if archer is not None and _rule_supporter_can_precede_current_ko_attack(observation, options, ARCHER):
                return archer
        ariana = _rule_find_supporter_option(observation, options, ARIANA)
        if ariana is not None and _rule_supporter_can_precede_current_ko_attack(observation, options, ARIANA):
            return ariana

    archer = _rule_find_supporter_option(observation, options, ARCHER)
    apollo_direct_score = _rule_apollo_direct_play_score(observation) if archer is not None else None
    if (
        archer is not None
        and apollo_direct_score is not None
        and apollo_direct_score >= 38_000
        and _rule_supporter_can_precede_current_ko_attack(observation, options, ARCHER)
    ):
        return archer

    ariana = _rule_find_supporter_option(observation, options, ARIANA)
    if (
        ariana is not None
        and _ariana_draw_count_for_player(player) > 0
        and _rule_supporter_can_precede_current_ko_attack(observation, options, ARIANA)
    ):
        return ariana

    if (
        archer is not None
        and apollo_direct_score is not None
        and apollo_direct_score >= 22_000
        and _rule_supporter_can_precede_current_ko_attack(observation, options, ARCHER)
    ):
        return archer

    if (
        petrel is not None
        and _rule_petrel_bridge_needed(observation)
        and _rule_supporter_can_precede_current_ko_attack(observation, options, PETREL)
    ):
        return petrel

    hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    deck_count = _deck_count(player)
    if archer is not None and (
        deck_count <= 4
        or _hand_count(player) <= 5
        or hand_supporters <= 1
        or not _rule_has_energy_in_hand(player)
    ) and _rule_supporter_can_precede_current_ko_attack(observation, options, ARCHER):
        return archer

    factory_enabler = _rule_find_factory_enabler_supporter(observation, options)
    if factory_enabler is not None:
        return factory_enabler

    return None


def _rule_find_opening_lance_search_item(observation, options):
    current, _, player = _current_player(observation)
    setup_incomplete = _rule_setup_incomplete(player)
    if not _proton_opening_allowed(current, player, setup_incomplete):
        return None
    if PROTON in _rule_hand_ids(player):
        return None
    for item_id in (TEAM_ROCKET_TRANSCEIVER, POKEGEAR):
        item = _rule_find_card_option(observation, options, item_id, (7, "play"))
        if item is not None:
            return item
    return None


def _rule_find_pre_support_factory_before_supporter_option(observation, options, supporter_index=None):
    current, _, player = _current_player(observation)
    if _supporter_played_this_turn(current, player):
        return None
    if _stadium_played_this_turn(current, player):
        return None
    if _rule_low_deck_draw_search_blocked(observation, options):
        return None
    if FACTORY in _rule_stadium_ids(current):
        return None
    if supporter_index is None:
        supporter_index = _rule_find_supporter_to_play(observation, options)
    if supporter_index is None:
        return None
    return _rule_find_card_option(observation, options, FACTORY, (7, "play"))


def _rule_find_post_support_poke_pad_option(observation, options):
    current, _, player = _current_player(observation)
    if not _supporter_played_this_turn(current, player):
        return None
    if _rule_low_deck_draw_search_blocked(observation, options):
        return None
    return _rule_find_card_option(observation, options, POKE_PAD, (7, "play"))


def _rule_find_post_support_factory_option(observation, options):
    current, _, player = _current_player(observation)
    if not _supporter_played_this_turn(current, player):
        return None
    if _rule_low_deck_draw_search_blocked(observation, options):
        return None
    ability = _rule_find_card_option(observation, options, FACTORY, (10, "ability"))
    if ability is not None:
        return ability
    if _stadium_played_this_turn(current, player):
        return None
    return _rule_find_card_option(observation, options, FACTORY, (7, "play"))


def _rule_find_post_support_rocket_feather_fuel_search_option(observation, options):
    current, _, player = _current_player(observation)
    if not _supporter_played_this_turn(current, player):
        return None
    if _rule_low_deck_draw_search_blocked(observation, options):
        return None
    if _rule_find_attack_option(options, ROCKET_FEATHER_ATTACK) is None:
        return None

    deck_supporters = _deck_card_count_for_policy(player, lambda card_id: card_id in ROCKET_SUPPORTERS)
    if deck_supporters <= 0 or _deck_count(player) <= 1:
        return None

    for item_id in (ROTO_STICK, POKEGEAR, TEAM_ROCKET_TRANSCEIVER):
        item = _rule_find_card_option(observation, options, item_id, (7, "play"))
        if item is not None:
            return item
    return None


def _rule_find_post_support_supporter_thinning_option(observation, options):
    for item_id in (ROTO_STICK, POKEGEAR, TEAM_ROCKET_TRANSCEIVER):
        item = _rule_find_card_option(observation, options, item_id, (7, "play"))
        if item is not None:
            return item
    return None


def _rule_post_support_rocket_feather_damage_shortage(observation, options):
    if _rule_find_attack_option(options, ROCKET_FEATHER_ATTACK) is None:
        return False
    shortage = _rule_rocket_feather_fuel_shortage(observation)
    return shortage is not None


def _rule_attack_condition_missing_before_factory(observation, options):
    if _rule_has_meaningful_main_attack_option(observation, options):
        return False
    if _rule_find_evolution_option(observation, options, HONCHKROW) is not None:
        return False
    if _rule_find_evolution_option(observation, options, PORYGON2) is not None:
        return False

    current, _, player = _current_player(observation)
    hand_ids = _rule_hand_ids(player)
    deck_ids = _deck_card_ids_for_policy(player)
    field_ids = _field_card_ids(player)

    missing_energy = (
        not _energy_attached_this_turn(current, player)
        and _needs_ariana_energy_dig(player)
    )
    missing_honchkrow = (
        MURKROW in field_ids
        and HONCHKROW not in hand_ids
        and HONCHKROW in deck_ids
    )
    missing_porygon2 = (
        PORYGON in field_ids
        and PORYGON2 not in hand_ids
        and PORYGON2 in deck_ids
        and (IGNITION_ENERGY in hand_ids or IGNITION_ENERGY in deck_ids)
    )
    return missing_energy or missing_honchkrow or missing_porygon2


def _rule_find_post_support_pre_factory_attack_condition_search_option(observation, options):
    current, _, player = _current_player(observation)
    if not _supporter_played_this_turn(current, player):
        return None
    if _rule_low_deck_draw_search_blocked(observation, options):
        return None
    if _rule_find_post_support_factory_option(observation, options) is None:
        return None
    if not _rule_attack_condition_missing_before_factory(observation, options):
        return None
    return _rule_find_post_support_supporter_thinning_option(observation, options)


RULE_SHALLOW_SEARCH_MAX_CANDIDATES = 8
RULE_SHALLOW_SEARCH_ITERATIONS = 22
RULE_SHALLOW_SEARCH_OVERRIDE_MARGIN = 8_200
RULE_SHALLOW_SEARCH_SELECTED_PRIOR = 18_000
RULE_TINY_NEURAL_SCALE = 7_800
RULE_TINY_NEURAL_HIDDEN_WEIGHTS = (
    (0.52, 1.20, 1.90, 0.42, 0.35, 0.25, 0.22, 0.25, -0.25, -0.10, 0.35, 0.55),
    (-0.18, 0.42, 0.80, 0.72, 0.65, 0.58, 0.50, 0.48, 0.20, 0.18, -0.10, -0.20),
    (0.24, -0.35, -0.20, 0.40, 0.46, 0.72, 0.60, 0.52, 0.50, 0.35, -0.12, 0.28),
    (0.10, -0.42, -0.75, 0.18, 0.10, 0.20, 0.18, 0.14, -0.55, -0.55, 0.40, 0.30),
)
RULE_TINY_NEURAL_HIDDEN_BIAS = (0.05, -0.18, -0.08, 0.12)
RULE_TINY_NEURAL_OUTPUT_WEIGHTS = (1.28, 0.78, 0.42, -0.44)
RULE_TINY_NEURAL_OUTPUT_BIAS = -0.03
def _rule_search_card_kind(identifier):
    if identifier in (TEAM_ROCKET_TRANSCEIVER, POKEGEAR, ROTO_STICK, POKE_PAD, NIGHT_STRETCHER, MIRACLE_HEADSET):
        return "search"
    if identifier == FACTORY:
        return "factory"
    if identifier in SUPPORTER_CARD_IDS:
        return "supporter"
    if identifier in (MURKROW, ARTICUNO, HONCHKROW, PORYGON, PORYGON2):
        return "pokemon"
    if identifier in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY):
        return "energy"
    return "other"


def _rule_low_deck_action_penalty(player, option_type, identifier):
    deck_count = _deck_count(player)
    critical_threshold = _policy_rule_number("loGuard", "criticalDeckThreshold", 4)
    near_threshold = _policy_rule_number("loGuard", "nearDeckThreshold", 6)
    if deck_count > near_threshold:
        return 0

    critical = deck_count <= critical_threshold
    if identifier == FACTORY:
        return _policy_rule_number(
            "loGuard",
            "criticalFactoryPenalty" if critical else "nearFactoryPenalty",
            260_000 if critical else 90_000,
        )
    if option_type in (7, "play") and identifier in (ROTO_STICK, POKEGEAR, TEAM_ROCKET_TRANSCEIVER, POKE_PAD):
        return _policy_rule_number(
            "loGuard",
            "criticalSearchPenalty" if critical else "nearSearchPenalty",
            120_000 if critical else 32_000,
        )
    if option_type in (7, "play") and identifier in SUPPORTER_CARD_IDS:
        drawish_supporter = identifier in (ARIANA, ARCHER, PETREL, PROTON)
        return _policy_rule_number(
            "loGuard",
            "criticalSupportPenalty" if critical else "nearSupportPenalty",
            130_000 if critical else 36_000,
        ) if drawish_supporter else 0
    return 0


def _rule_low_deck_constructive_action_index(observation, options):
    _, _, player = _current_player(observation)
    alakazam_recovery = _rule_find_alakazam_articuno_recovery_option(observation, options)
    if alakazam_recovery is not None:
        return alakazam_recovery

    alakazam_escape = _rule_find_alakazam_porygon_escape_option(observation, options)
    if alakazam_escape is not None:
        return alakazam_escape

    if _needs_seed_out_bench_guard(player):
        basic = _rule_find_alakazam_basic_play_option(observation, options)
        if basic is None:
            basic = _rule_find_dragapult_basic_play_option(observation, options)
        if basic is None:
            basic = _rule_find_basic_play_option(observation, options, (MURKROW, PORYGON))
        if basic is not None:
            return basic

    miracle_high_prize_ko = _rule_find_miracle_headset_high_prize_ko_option(observation, options)
    if miracle_high_prize_ko is not None:
        return miracle_high_prize_ko

    immediate_attack = _rule_find_immediate_ko_attack_option(observation, options)
    if immediate_attack is not None:
        return immediate_attack

    for evolution_id in (HONCHKROW, PORYGON2):
        evolution = _rule_find_evolution_option(observation, options, evolution_id)
        if evolution is not None:
            return evolution

    attach = _rule_find_attach_option(observation, options)
    if attach is not None:
        return attach

    if not _opponent_active_has_hop_dodge_protection(observation):
        attack = _rule_choose_best_main_attack(observation, options)
        if attack is not None:
            return attack

    murkrow_ko_attack = _rule_find_murkrow_ko_attack_option(observation, options)
    if murkrow_ko_attack is not None:
        return murkrow_ko_attack

    if not _opponent_active_has_hop_dodge_protection(observation):
        tempo_attack = _rule_find_tempo_attack_option(observation, options)
        if tempo_attack is not None:
            return tempo_attack

    basic = _rule_find_alakazam_basic_play_option(observation, options)
    if basic is None:
        basic = _rule_find_basic_play_option(observation, options, (MURKROW, PORYGON))
    return basic


def _rule_low_deck_draw_search_blocked(observation, options=None):
    if _alakazam_taunt_lock_ready(observation):
        return True

    _, _, player = _current_player(observation)
    deck_count = _deck_count(player)
    critical_threshold = _policy_rule_number("loGuard", "criticalDeckThreshold", 4)
    near_threshold = _policy_rule_number("loGuard", "nearDeckThreshold", 6)
    alakazam_threshold = _policy_rule_number("alakazamLockPlan", "lowDeckReleaseThreshold", 8)
    alakazam_preserve_threshold = _policy_rule_number("alakazamLockPlan", "deckPreserveThreshold", 20)

    if deck_count <= critical_threshold:
        return True
    if _alakazam_lock_deck_preservation_active(observation) and deck_count <= alakazam_preserve_threshold:
        return True
    if _alakazam_resist_veil_plan_active(observation) and deck_count <= alakazam_threshold:
        return True
    if options is not None and deck_count <= near_threshold:
        return _rule_low_deck_constructive_action_index(observation, options) is not None
    return False


def _rule_low_deck_finish_action_index(observation, options):
    if not _rule_low_deck_draw_search_blocked(observation, options):
        return None

    constructive = _rule_low_deck_constructive_action_index(observation, options)
    if constructive is not None:
        return constructive

    for option_index, option in enumerate(options):
        if _option_type(option) in (14, "end"):
            return option_index
    return None


def _rule_hard_main_action_index(observation, options):
    select = _read(observation, "select", {})
    if _select_context(select) not in (0, "main"):
        return None
    current, _, player = _current_player(observation)
    if not _supporter_played_this_turn(current, player):
        giovanni_lethal = _rule_find_supporter_option(observation, options, GIOVANNI)
        if giovanni_lethal is not None and _sakaki_can_take_remaining_prizes(observation, giovanni_from_hand=True):
            return giovanni_lethal
    alakazam_recovery = _rule_find_alakazam_articuno_recovery_option(observation, options)
    if alakazam_recovery is not None:
        return alakazam_recovery
    alakazam_escape = _rule_find_alakazam_porygon_escape_option(observation, options)
    if alakazam_escape is not None:
        return alakazam_escape
    if _needs_seed_out_bench_guard(player):
        basic = _rule_find_alakazam_basic_play_option(observation, options)
        if basic is None:
            basic = _rule_find_dragapult_basic_play_option(observation, options)
        if basic is None:
            basic = _rule_find_basic_play_option(observation, options, (MURKROW, PORYGON))
        if basic is not None:
            return basic
    low_deck_finish = _rule_low_deck_finish_action_index(observation, options)
    if low_deck_finish is not None:
        return low_deck_finish
    alakazam_basic = _rule_find_alakazam_basic_play_option(observation, options)
    if alakazam_basic is not None:
        return alakazam_basic
    dragapult_basic = _rule_find_dragapult_basic_play_option(observation, options)
    if dragapult_basic is not None:
        return dragapult_basic
    if not _supporter_played_this_turn(current, player):
        board_first = _rule_find_evolution_option(observation, options, HONCHKROW)
        if board_first is None:
            board_first = _rule_find_evolution_option(observation, options, PORYGON2)
        if board_first is None:
            board_first = _rule_find_basic_play_option(observation, options, (MURKROW, PORYGON))
        if board_first is None:
            pre_support_attach = _rule_find_pre_support_attach_option(observation, options)
            if pre_support_attach is not None:
                return pre_support_attach
    return None


def _rule_attack_score_for_option(observation, option):
    attack_id = _attack_id(option)
    if _alakazam_lock_strategy_active(observation) and attack_id not in RULE_ONLY_MURKROW_KO_ATTACKS:
        return -1_000_000
    if attack_id == ROCKET_FEATHER_ATTACK:
        return _rule_attack_efficiency_score(observation, ROCKET_FEATHER_ATTACK)
    if attack_id in PORYGON2_R_COMMAND_ATTACKS:
        return _rule_attack_efficiency_score(observation, next(iter(PORYGON2_R_COMMAND_ATTACKS)))
    if attack_id in RULE_ONLY_MURKROW_KO_ATTACKS:
        murkrow_ko = _rule_find_murkrow_ko_attack_option(observation, _select_options(_read(observation, "select", {})))
        return 80_000 if murkrow_ko is not None else -42_000
    if attack_id == MURKROW_TEMPT_ATTACK:
        _, _, player = _current_player(observation)
        hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)
        return 12_000 if hand_supporters <= 0 else -34_000
    return -18_000


def _rule_option_enables_attack_score(observation, option):
    option_type = _option_type(option)
    identifier = _rule_option_id(observation, option)
    target = _target_card_from_option(observation, option)
    target_id = _card_id(target)
    current, _, player = _current_player(observation)
    opponent_target = _opponent_active_card(observation)
    remaining_hp = _remaining_hp(opponent_target)
    if remaining_hp <= 0:
        return 0

    attacker_id = None
    if option_type in (9, "evolve"):
        attacker_id = identifier
    elif option_type in (8, "attach"):
        attacker_id = target_id
    elif option_type in (7, "play") and identifier in (MURKROW, PORYGON):
        attacker_id = identifier

    if attacker_id == HONCHKROW:
        hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)
        damage = _rocket_feather_damage_per_supporter(opponent_target) * hand_supporters
        if damage >= remaining_hp and hand_supporters > 0:
            return 82_000 + _prize_count_for_knockout(opponent_target) * 28_000
        if damage > 0:
            return min(34_000, damage * 140)
    if attacker_id == PORYGON2:
        discard_supporters = _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS)
        damage = _damage_after_type_modifier(
            discard_supporters * _policy_rule_number("porygon2RCommandFallback", "damagePerTrashRocketSupporter", 20),
            opponent_target,
            "colorless",
        )
        if damage >= remaining_hp and discard_supporters > 0:
            return 78_000 + _prize_count_for_knockout(opponent_target) * 25_000
        if damage > 0:
            return min(28_000, damage * 120)
    if attacker_id == MURKROW and HONCHKROW in _rule_hand_ids(player):
        return 14_000
    if attacker_id == PORYGON and PORYGON2 in _rule_hand_ids(player):
        return 10_000

    if option_type in (7, "play") and identifier == FACTORY and _supporter_played_this_turn(current, player):
        return 18_000
    if option_type in (10, "ability") and identifier == FACTORY:
        return 20_000
    return 0


def _rule_needs_energy_or_evolution(player):
    field_ids = _field_card_ids(player)
    hand_ids = _rule_hand_ids(player)
    no_energy = not _rule_has_energy_in_hand(player)
    needs_honchkrow = MURKROW in field_ids and HONCHKROW not in hand_ids
    needs_porygon2 = PORYGON in field_ids and PORYGON2 not in hand_ids and _porygon_development_allowed(player)
    return no_energy or needs_honchkrow or needs_porygon2


def _rule_tiny_neural_features(observation, option):
    option_type = _option_type(option)
    identifier = _rule_option_id(observation, option)
    current, _, player = _current_player(observation)
    target = _opponent_active_card(observation)
    remaining_hp = max(0, _remaining_hp(target))
    prize_count = _prize_count_for_knockout(target) if target is not None else 1
    is_attack = 1.0 if option_type in (13, "attack") else 0.0
    attack_score = _rule_attack_score_for_option(observation, option) if is_attack else 0.0
    kind = _rule_search_card_kind(identifier)
    return (
        1.0,
        is_attack,
        1.0 if is_attack and attack_score >= 200_000 else 0.0,
        1.0 if option_type in (9, "evolve") else 0.0,
        1.0 if option_type in (8, "attach") else 0.0,
        1.0 if kind == "supporter" else 0.0,
        1.0 if kind == "search" else 0.0,
        1.0 if kind == "factory" or option_type in (10, "ability") else 0.0,
        min(1.0, _hand_count(player) / 10.0),
        min(1.0, _deck_count(player) / 35.0),
        min(1.0, _bench_top_count(player) / max(1, _bench_limit(player))),
        min(1.0, prize_count / 3.0) if remaining_hp > 0 else 0.0,
    )


def _rule_tiny_neural_score(observation, option):
    features = _rule_tiny_neural_features(observation, option)
    hidden_values = []
    for weights, bias in zip(RULE_TINY_NEURAL_HIDDEN_WEIGHTS, RULE_TINY_NEURAL_HIDDEN_BIAS):
        activation = bias + sum(weight * feature for weight, feature in zip(weights, features))
        hidden_values.append(max(0.0, activation))
    output = RULE_TINY_NEURAL_OUTPUT_BIAS + sum(
        weight * value for weight, value in zip(RULE_TINY_NEURAL_OUTPUT_WEIGHTS, hidden_values)
    )
    return int(output * RULE_TINY_NEURAL_SCALE)


def _rule_shallow_retreat_score(observation, option, selected_bonus=0):
    current, _, player = _current_player(observation)
    active_card = _top_card(_read(player, "active", []))
    active_id = _card_id(active_card)
    active_energy_cards = _attached_energy_cards(active_card)
    target = _target_card_from_option(observation, option)
    promotion_card = _promotion_card_from_option(observation, option)
    turn_plan = _build_donkrow_turn_plan(observation)

    if (
        turn_plan is not None
        and not turn_plan.seed_guard_blocked
        and turn_plan.needs_switch
        and _switch_option_targets_plan_attacker(option, target, promotion_card, turn_plan)
    ):
        return selected_bonus + 72_000 + max(0, turn_plan.score // 30)

    ready_bench_attackers = _ready_bench_attacker_ids(player)
    has_attack_option = _has_any_main_attack_option(observation)
    score = selected_bonus
    if _alakazam_porygon_active_escape_needed(observation):
        promotion_card = _promotion_card_from_option(observation, option)
        promotion_id = _card_id(promotion_card)
        if promotion_id == MURKROW or any(_card_id(card) == MURKROW for card in _field_top_cards(player)):
            return score + 180_000

    if ready_bench_attackers:
        score += 14_000
        if HONCHKROW in ready_bench_attackers:
            score += 16_000
        if PORYGON2 in ready_bench_attackers:
            score += 10_000
        if active_energy_cards > 0:
            score -= 28_000 + active_energy_cards * 16_000
        if has_attack_option:
            score -= 38_000
        return score

    score -= 96_000
    if active_energy_cards > 0:
        score -= 90_000 + active_energy_cards * 24_000
    if active_id in (MURKROW, HONCHKROW, PORYGON2):
        score -= 20_000
    if not has_attack_option:
        score -= 12_000
    return score


def _rule_option_flow_family(observation, option):
    option_type = _option_type(option)
    identifier = _rule_option_id(observation, option)
    if option_type in (14, "end"):
        return "end"
    if option_type in (13, "attack"):
        return "attack"
    if option_type in (12, "retreat", "switch", "promote"):
        return "switch"
    if option_type in (9, "evolve"):
        return "evolve"
    if option_type in (8, "attach"):
        return "attach"
    if option_type in (10, "ability") and identifier == FACTORY:
        return "factory"
    if option_type in (7, "play"):
        kind = _rule_search_card_kind(identifier)
        if kind == "pokemon":
            return "basic"
        if kind in ("supporter", "search", "factory"):
            return kind
        if kind == "energy":
            return "energy"
    return "other"


def _rule_miracle_headset_main_phase_allowed(observation, options, option_index):
    if _rule_find_card_option(observation, options, MIRACLE_HEADSET, (7, "play")) != option_index:
        return False
    if _rule_find_miracle_headset_high_prize_ko_option(observation, options) == option_index:
        return True
    if _rule_find_miracle_headset_athena_rescue_option(observation, options) == option_index:
        return True

    # Side-1 endgame KO is still a real use case, but generic Proton recovery is not.
    current, _, player = _current_player(observation)
    snapshot = _rule_rocket_feather_ko_snapshot(observation)
    if (
        snapshot is not None
        and snapshot["prize_count"] >= max(1, _prize_cards_remaining(player))
        and _rule_rocket_feather_ko_reachable_after_miracle(observation, min_prize=1)
        and _rule_find_attack_option(options, ROCKET_FEATHER_ATTACK) is not None
    ):
        return True
    return False


def _rule_search_candidate_allowed_by_flow(observation, options, option_index):
    option = options[option_index]
    identifier = _rule_option_id(observation, option)
    if identifier == MIRACLE_HEADSET:
        return _rule_miracle_headset_main_phase_allowed(observation, options, option_index)
    if identifier == POKE_PAD:
        return _best_poke_pad_target_score(observation) > 0
    if identifier == NIGHT_STRETCHER:
        return _best_night_stretcher_target_score(observation) > 0
    if identifier in (TEAM_ROCKET_TRANSCEIVER, POKEGEAR, ROTO_STICK):
        if _alakazam_articuno_petrel_recovery_needed(observation):
            return True
        return (
            _rule_find_post_support_rocket_feather_fuel_search_option(observation, options) == option_index
            or _rule_find_post_support_pre_factory_attack_condition_search_option(observation, options) == option_index
            or _rule_find_opening_lance_search_item(observation, options) == option_index
            or _dragapult_articuno_search_needed(observation)
            or _rule_needs_energy_or_evolution(_current_player(observation)[2])
        )
    return True


def _rule_supporter_candidate_allowed_by_flow(observation, option):
    identifier = _rule_option_id(observation, option)
    current, _, player = _current_player(observation)
    if identifier == PROTON:
        return _rule_lance_opening_allowed_for_player(current, player) or _dragapult_articuno_search_needed(observation)
    if identifier == GIOVANNI:
        return (
            _sakaki_can_take_remaining_prizes(observation, giovanni_from_hand=True)
            or _sakaki_prize_race_ko_score(observation, giovanni_from_hand=True) is not None
            or _sakaki_hop_dodge_escape_ko_score(observation, giovanni_from_hand=True) is not None
            or _alakazam_porygon_active_escape_needed(observation)
        )
    if identifier == PETREL and _alakazam_articuno_petrel_recovery_needed(observation):
        return True
    return identifier in (ARIANA, ARCHER, PETREL)


def _rule_flow_candidate_allowed(observation, options, selected_index, option_index):
    if option_index == selected_index:
        return True
    if not isinstance(selected_index, int) or not (0 <= selected_index < len(options)):
        return False
    if not isinstance(option_index, int) or not (0 <= option_index < len(options)):
        return False

    selected = options[selected_index]
    option = options[option_index]
    selected_family = _rule_option_flow_family(observation, selected)
    candidate_family = _rule_option_flow_family(observation, option)

    # The flow's "end turn" decision means every required flow item has already
    # declined.  Neural/MCTS must not turn that into generic card spending.
    if selected_family == "end":
        return False
    if candidate_family == "end":
        return False
    if selected_family != candidate_family:
        return False
    if candidate_family == "evolve" and _alakazam_forbidden_evolution_option(observation, option):
        return False
    if candidate_family == "evolve" and _dragapult_forbidden_evolution_option(observation, option):
        return False
    if candidate_family == "basic" and _dragapult_forbidden_basic_play_option(observation, option):
        return False
    if candidate_family == "attach":
        return not _rule_forbidden_energy_attach_option(observation, option)
    if candidate_family == "switch":
        return _rule_shallow_retreat_score(observation, option, 0) > 0
    if candidate_family == "attack":
        return _rule_attack_score_for_option(observation, option) > -500_000
    if candidate_family == "search":
        return _rule_search_candidate_allowed_by_flow(observation, options, option_index)
    if candidate_family == "supporter":
        return _rule_supporter_candidate_allowed_by_flow(observation, option)
    return True


def _rule_shallow_prior_score(observation, options, option_index, selected_index=None):
    if option_index is None or not (0 <= option_index < len(options)):
        return -1_000_000
    option = options[option_index]
    option_type = _option_type(option)
    identifier = _rule_option_id(observation, option)
    current, _, player = _current_player(observation)
    supporter_played = _supporter_played_this_turn(current, player)
    score = RULE_SHALLOW_SEARCH_SELECTED_PRIOR if option_index == selected_index else 0

    if option_type in (8, "attach") and _rule_forbidden_energy_attach_option(observation, option):
        return -1_000_000
    if option_type in (9, "evolve") and _alakazam_forbidden_evolution_option(observation, option):
        return -1_000_000
    if option_type in (9, "evolve") and _dragapult_forbidden_evolution_option(observation, option):
        return -1_000_000
    if option_type in (7, "play") and _dragapult_forbidden_basic_play_option(observation, option):
        return -1_000_000

    hard = _rule_hard_main_action_index(observation, options)
    if hard is not None:
        return 1_000_000 if option_index == hard else -1_000_000
    if (
        option_type in (7, "play")
        and identifier == MIRACLE_HEADSET
        and not _rule_miracle_headset_main_phase_allowed(observation, options, option_index)
    ):
        return -1_000_000

    if option_type in (14, "end"):
        return -55_000
    if option_type in (13, "attack"):
        score += _rule_attack_score_for_option(observation, option)
    elif option_type in (9, "evolve"):
        score += 58_000 if identifier == HONCHKROW else 42_000 if identifier == PORYGON2 else 0
        score += _rule_option_enables_attack_score(observation, option)
    elif option_type in (8, "attach"):
        if _rule_find_attach_option(observation, options) == option_index:
            score += 44_000
        if _rule_find_pre_support_attach_option(observation, options) == option_index:
            score += 18_000
        if _rule_find_opening_team_rocket_attach_option(observation, options) == option_index:
            score += 26_000
        score += _rule_option_enables_attack_score(observation, option)
    elif option_type in (10, "ability") and identifier == FACTORY:
        score += 48_000 if supporter_played else 18_000
    elif option_type in (12, "retreat", "switch", "promote"):
        return _rule_shallow_retreat_score(observation, option, score)
    elif option_type in (7, "play"):
        kind = _rule_search_card_kind(identifier)
        alakazam_score = _alakazam_basic_target_score(observation, identifier)
        if alakazam_score is not None:
            score += alakazam_score
        else:
            dragapult_score = _dragapult_basic_target_score(observation, identifier)
            if dragapult_score is not None:
                score += dragapult_score
            elif identifier in (MURKROW, ARTICUNO, PORYGON):
                score += 34_000 + _rule_option_enables_attack_score(observation, option)
            elif identifier in (HONCHKROW, PORYGON2):
                score += 20_000 + _rule_option_enables_attack_score(observation, option)
        if kind == "supporter":
            supporter = _rule_find_supporter_to_play(observation, options)
            if supporter == option_index:
                score += 54_000
            if identifier == ARIANA:
                score += max(0, _ariana_draw_count_for_player(player)) * 7_500
            elif identifier == ARCHER:
                apollo_score = _rule_apollo_direct_play_score(observation)
                if apollo_score is not None:
                    score += min(70_000, apollo_score)
            elif identifier == PETREL and _rule_petrel_bridge_needed(observation):
                score += 24_000
                if _dragapult_petrel_poke_pad_attack_bridge_needed(observation):
                    score += 58_000
            elif identifier == GIOVANNI:
                sakaki_score = _sakaki_prize_race_ko_score(observation, giovanni_from_hand=True)
                if sakaki_score is not None or _sakaki_can_take_remaining_prizes(observation, giovanni_from_hand=True):
                    score += 72_000
                else:
                    score -= 110_000
            elif identifier == PROTON:
                score += 18_000 if (
                    _rule_lance_opening_allowed_for_player(current, player)
                    or _dragapult_articuno_search_needed(observation)
                ) else -24_000
        elif kind == "factory":
            score += 44_000 if _rule_find_pre_support_factory_before_supporter_option(observation, options) == option_index else 18_000
        elif kind == "search":
            if _rule_find_post_support_rocket_feather_fuel_search_option(observation, options) == option_index:
                score += 42_000
            if _rule_find_post_support_pre_factory_attack_condition_search_option(observation, options) == option_index:
                score += 40_000
            if _rule_find_opening_lance_search_item(observation, options) == option_index:
                score += 32_000
            if identifier in (TEAM_ROCKET_TRANSCEIVER, POKEGEAR, ROTO_STICK) and _dragapult_articuno_search_needed(observation):
                score += 48_000
            if identifier == POKE_PAD:
                score += 22_000
            if identifier == NIGHT_STRETCHER:
                score += 18_000 if _needs_seed_out_bench_guard(player) else 8_000
            if identifier in (TEAM_ROCKET_TRANSCEIVER, POKEGEAR, ROTO_STICK) and _rule_needs_energy_or_evolution(player):
                score += 12_000
            if identifier == MIRACLE_HEADSET and _rule_rocket_feather_ko_reachable_after_miracle(observation, min_prize=2):
                score += 76_000

    if _rule_board_collapse_reset_needed(observation):
        if option_type in (7, "play") and identifier in (ARCHER, PROTON, MURKROW, PORYGON, POKE_PAD, NIGHT_STRETCHER):
            score += 18_000
        if option_type in (13, "attack"):
            score -= 10_000
    if _deck_count(player) <= 4 and option_type in (7, "play") and identifier in (ARIANA, ARCHER, POKEGEAR, ROTO_STICK):
        score -= 28_000
    score -= _rule_low_deck_action_penalty(player, option_type, identifier)

    score += _rule_option_enables_attack_score(observation, option) // 2
    score += _rule_tiny_neural_score(observation, option)
    return score


def _rule_second_ply_estimate(observation, option):
    option_type = _option_type(option)
    identifier = _rule_option_id(observation, option)
    current, _, player = _current_player(observation)
    score = 0
    if option_type in (7, "play") and identifier == ARIANA:
        score += max(0, _ariana_draw_count_for_player(player)) * 6_000
        if FACTORY not in _rule_stadium_ids(current):
            score += 7_500
    if option_type in (7, "play") and identifier == ARCHER:
        apollo_score = _rule_apollo_direct_play_score(observation)
        if apollo_score is not None:
            score += min(22_000, apollo_score // 2)
    if option_type in (7, "play") and identifier == PETREL:
        if FACTORY not in _rule_stadium_ids(current):
            score += 9_000
        if _alakazam_petrel_poke_pad_bridge_needed(observation):
            score += 16_000
        if _rule_petrel_poke_pad_bridge_needed(player):
            score += 8_000
        if _rule_petrel_energy_bridge_needed(player):
            score += 8_000
        if _dragapult_petrel_poke_pad_attack_bridge_needed(observation):
            score += 16_000
    if option_type in (7, "play") and identifier in (TEAM_ROCKET_TRANSCEIVER, POKEGEAR, ROTO_STICK):
        score += 8_000 if _supporter_energy_dig_preference(current, player) is not None else 3_500
    if option_type in (7, "play") and identifier == POKE_PAD:
        deck_ids = _deck_card_ids_for_policy(player)
        if HONCHKROW in deck_ids and MURKROW in _field_card_ids(player):
            score += 11_000
        if PORYGON2 in deck_ids and PORYGON in _field_card_ids(player):
            score += 8_000
        if _needs_seed_out_bench_guard(player) and any(card_id in BASIC_SETUP_POKEMON for card_id in deck_ids):
            score += 16_000
    if option_type in (7, "play") and identifier == NIGHT_STRETCHER:
        score += 18_000 if _needs_seed_out_bench_guard(player) else 5_000
    if option_type in (8, "attach", 9, "evolve"):
        score += _rule_option_enables_attack_score(observation, option) // 2
    if option_type in (13, "attack"):
        score += _rule_attack_score_for_option(observation, option) // 4
    score -= _rule_low_deck_action_penalty(player, option_type, identifier) // 2
    return score


def _rule_shallow_rollout_value(observation, options, option_index, selected_index, depth=2):
    option = options[option_index]
    value = _rule_shallow_prior_score(observation, options, option_index, selected_index)
    if depth >= 2:
        value += _rule_second_ply_estimate(observation, option)
    return value


def _rule_main_candidate_pool(observation, options, selected_index):
    scored = []
    for option_index, option in enumerate(options):
        if not _rule_flow_candidate_allowed(observation, options, selected_index, option_index):
            continue
        if _option_type(option) in (8, "attach") and _rule_forbidden_energy_attach_option(observation, option):
            continue
        score = _rule_shallow_prior_score(observation, options, option_index, selected_index)
        scored.append((score, -option_index, option_index))
    scored.sort(reverse=True)
    result = []
    if isinstance(selected_index, int) and 0 <= selected_index < len(options):
        result.append(selected_index)
    for _, _, option_index in scored:
        if option_index not in result:
            result.append(option_index)
        if len(result) >= RULE_SHALLOW_SEARCH_MAX_CANDIDATES:
            break
    return result


def _rule_shallow_mcts_stats(observation, options, candidate_indices, selected_index):
    stats = {}
    for option_index in candidate_indices:
        prior = _rule_shallow_prior_score(observation, options, option_index, selected_index)
        stats[option_index] = {"visits": 1, "value": float(prior), "prior": float(prior)}
    for iteration in range(RULE_SHALLOW_SEARCH_ITERATIONS):
        total_visits = sum(item["visits"] for item in stats.values())
        best_index = None
        best_ucb = None
        for option_index, item in stats.items():
            mean = item["value"] / max(1, item["visits"])
            exploration = 1_600.0 * (((total_visits + 1.0) / (item["visits"] + 1.0)) ** 0.5)
            ucb = mean + exploration + item["prior"] * 0.015
            if best_ucb is None or ucb > best_ucb:
                best_ucb = ucb
                best_index = option_index
        if best_index is None:
            break
        depth = 2 if iteration % 2 == 0 else 1
        rollout = _rule_shallow_rollout_value(observation, options, best_index, selected_index, depth=depth)
        stats[best_index]["visits"] += 1
        stats[best_index]["value"] += float(rollout)
    return stats


def _rule_shallow_mcts_scores(observation, options, candidate_indices, selected_index):
    stats = _rule_shallow_mcts_stats(observation, options, candidate_indices, selected_index)
    return {option_index: item["value"] / max(1, item["visits"]) for option_index, item in stats.items()}


def _rule_shallow_search_decision(observation, options, selected_index):
    decision = {
        "coreIndex": selected_index,
        "finalIndex": selected_index,
        "hardProtected": False,
        "override": False,
        "overrideMargin": RULE_SHALLOW_SEARCH_OVERRIDE_MARGIN,
        "candidateLimit": RULE_SHALLOW_SEARCH_MAX_CANDIDATES,
        "iterations": RULE_SHALLOW_SEARCH_ITERATIONS,
        "reason": "固定ルール候補をそのまま採用",
        "candidateIndices": [],
        "scores": {},
        "stats": {},
    }
    if not isinstance(selected_index, int) or not (0 <= selected_index < len(options)):
        decision["reason"] = "固定ルール候補が無効なため補正なし"
        return decision
    hard = _rule_hard_main_action_index(observation, options)
    if hard is not None:
        decision.update(
            {
                "finalIndex": hard,
                "hardProtected": True,
                "hardIndex": hard,
                "reason": "KO担保、サカキリーサル、種切れ回避、サポート前の盤面/手張りなどの固定保護ルールを優先",
            }
        )
        return decision
    candidate_indices = _rule_main_candidate_pool(observation, options, selected_index)
    decision["candidateIndices"] = candidate_indices
    if len(candidate_indices) < 2:
        decision["reason"] = "比較候補が1個以下のため補正なし"
        return decision
    stats = _rule_shallow_mcts_stats(observation, options, candidate_indices, selected_index)
    scores = {option_index: item["value"] / max(1, item["visits"]) for option_index, item in stats.items()}
    selected_score = scores.get(selected_index, _rule_shallow_prior_score(observation, options, selected_index, selected_index))
    best_index = max(candidate_indices, key=lambda index: (scores.get(index, -1_000_000), -index))
    best_score = scores.get(best_index, -1_000_000)
    final_index = selected_index
    override = False
    if best_index != selected_index and best_score >= selected_score + RULE_SHALLOW_SEARCH_OVERRIDE_MARGIN:
        final_index = best_index
        override = True
    decision.update(
        {
            "finalIndex": final_index,
            "bestIndex": best_index,
            "selectedScore": selected_score,
            "bestScore": best_score,
            "scoreGap": best_score - selected_score,
            "override": override,
            "scores": scores,
            "stats": stats,
            "reason": "軽量先読み補正で置き換え" if override else "固定ルール候補との差が閾値未満のため補正なし",
        }
    )
    return decision


def _rule_shallow_search_rerank_main_action(observation, options, selected_index):
    return _rule_shallow_search_decision(observation, options, selected_index).get("finalIndex", selected_index)


RULE_TINY_NEURAL_FEATURE_NAMES = (
    "bias",
    "isAttack",
    "isKoLikeAttack",
    "isEvolution",
    "isAttach",
    "isSupporter",
    "isSearchCard",
    "isFactoryOrAbility",
    "handRatio",
    "deckRatio",
    "benchRatio",
    "opponentPrizeRatio",
)
RULE_TRACE_STATIC_EMITTED = False
RULE_TRACE_OPTIONAL_CANDIDATE_LIMIT = 8


def _rule_trace_number(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 3)
    return value


def _rule_trace_option_kind(option, identifier):
    option_type = _option_type(option)
    if option_type in (13, "attack"):
        return "attack"
    if option_type in (8, "attach"):
        return "attach"
    if option_type in (9, "evolve"):
        return "evolve"
    if option_type in (10, "ability"):
        return "ability"
    if option_type in (14, "end"):
        return "end"
    if option_type in (7, "play"):
        return _rule_search_card_kind(identifier)
    return str(option_type)


def _rule_trace_card_summary(card):
    if card is None:
        return None
    return {
        "id": _card_id(card),
        "name": _card_name_text(card),
        "hpRemaining": _remaining_hp(card),
        "prizesIfKo": _prize_count_for_knockout(card),
    }


def _rule_trace_state_snapshot(observation):
    current, opponent, player = _current_player(observation)
    your_index = _safe_int(_read(current, "yourIndex"), 0)
    opponent_index = 1 - your_index
    active = _active_card_for_player(current, your_index)
    opponent_active = _active_card_for_player(current, opponent_index)
    return {
        "step": _safe_int(_read(observation, "step"), 0),
        "turn": _safe_int(_read(current, "turn"), 0),
        "turnActionCount": _safe_int(_read(current, "turnActionCount"), 0),
        "yourIndex": your_index,
        "firstPlayer": _safe_int(_read(current, "firstPlayer"), -1),
        "supporterPlayed": _supporter_played_this_turn(current, player),
        "stadiumPlayed": _stadium_played_this_turn(current, player),
        "energyAttached": _energy_attached_this_turn(current, player),
        "handCount": _hand_count(player),
        "deckCount": _deck_count(player),
        "discardCount": _zone_count(player, "discard"),
        "benchCount": _bench_top_count(player),
        "handRocketSupporters": _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS),
        "discardRocketSupporters": _count_cards(player, ("discard",), lambda card_id: card_id in ROCKET_SUPPORTERS),
        "active": _rule_trace_card_summary(active),
        "opponentActive": _rule_trace_card_summary(opponent_active),
    }


def _rule_trace_neural_features(observation, option):
    features = _rule_tiny_neural_features(observation, option)
    return {
        name: _rule_trace_number(value)
        for name, value in zip(RULE_TINY_NEURAL_FEATURE_NAMES, features)
    }


def _rule_trace_option_scores(observation, options, option_index, selected_index, decision):
    option = options[option_index]
    identifier = _rule_option_id(observation, option)
    attack_id = _attack_id(option)
    stats = decision.get("stats", {}).get(option_index, {})
    scores = decision.get("scores", {})
    prior = stats.get("prior")
    if prior is None:
        prior = _rule_shallow_prior_score(observation, options, option_index, selected_index)
    visits = _safe_int(stats.get("visits"), 0)
    value = stats.get("value")
    mean_score = scores.get(option_index)
    if mean_score is None and visits:
        mean_score = value / max(1, visits)
    include_feature_breakdown = option_index in (
        selected_index,
        decision.get("finalIndex"),
        decision.get("bestIndex"),
        decision.get("hardIndex"),
    )
    trace = {
        "index": option_index,
        "labelJa": _rule_option_label_ja(observation, option),
        "kind": _rule_trace_option_kind(option, identifier),
        "optionType": _option_type(option),
        "cardId": identifier,
        "targetCardId": _rule_target_id(observation, option),
        "attackId": attack_id,
        "isFixedRuleCandidate": option_index == selected_index,
        "isFinalCandidate": option_index == decision.get("finalIndex"),
        "isBestSearchCandidate": option_index == decision.get("bestIndex"),
        "priorScore": _rule_trace_number(prior),
        "meanSearchScore": _rule_trace_number(mean_score),
        "visits": visits,
        "totalValue": _rule_trace_number(value),
        "secondPlyEstimate": _rule_second_ply_estimate(observation, option),
        "tinyNeuralScore": _rule_tiny_neural_score(observation, option),
        "attackScore": _rule_attack_score_for_option(observation, option) if _option_type(option) in (13, "attack") else 0,
        "enablesAttackScore": _rule_option_enables_attack_score(observation, option),
    }
    if include_feature_breakdown:
        trace["neuralFeatures"] = _rule_trace_neural_features(observation, option)
    return trace


def _rule_main_decision_trace(observation, options, selected_indices):
    core_index = _rule_choose_main_action_core(observation, options)
    decision = _rule_shallow_search_decision(observation, options, core_index)
    final_index = decision.get("finalIndex")
    actual_index = selected_indices[0] if selected_indices else final_index
    candidate_indices = list(decision.get("candidateIndices") or [])
    for index in (core_index, final_index, actual_index, decision.get("bestIndex"), decision.get("hardIndex")):
        if isinstance(index, int) and 0 <= index < len(options) and index not in candidate_indices:
            candidate_indices.append(index)
    candidate_details = [
        _rule_trace_option_scores(observation, options, option_index, core_index, decision)
        for option_index in candidate_indices
        if isinstance(option_index, int) and 0 <= option_index < len(options)
    ]
    candidate_details.sort(
        key=lambda item: (
            item["isFinalCandidate"],
            item["isFixedRuleCandidate"],
            item["meanSearchScore"] if item["meanSearchScore"] is not None else -1_000_000,
            -item["index"],
        ),
        reverse=True,
    )
    return {
        "type": "main",
        "traceVersion": 2,
        "state": _rule_trace_state_snapshot(observation),
        "actualSelectedIndex": actual_index,
        "fixedRuleIndex": core_index,
        "finalIndex": final_index,
        "bestSearchIndex": decision.get("bestIndex"),
        "hardProtected": decision.get("hardProtected", False),
        "override": decision.get("override", False),
        "overrideMargin": decision.get("overrideMargin"),
        "scoreGap": _rule_trace_number(decision.get("scoreGap")),
        "selectedScore": _rule_trace_number(decision.get("selectedScore")),
        "bestScore": _rule_trace_number(decision.get("bestScore")),
        "candidateLimit": decision.get("candidateLimit"),
        "iterations": decision.get("iterations"),
        "reasonJa": decision.get("reason"),
        "candidates": candidate_details,
    }


def _rule_optional_decision_trace(observation, options, selected_indices):
    select = _read(observation, "select", {})
    min_count, max_count = _selection_bounds(select)
    candidate_indices = list(range(min(len(options), RULE_TRACE_OPTIONAL_CANDIDATE_LIMIT)))
    for index in selected_indices:
        if isinstance(index, int) and 0 <= index < len(options) and index not in candidate_indices:
            candidate_indices.append(index)
    return {
        "type": "optional",
        "traceVersion": 2,
        "state": _rule_trace_state_snapshot(observation),
        "context": _select_context(select),
        "effectId": _rule_effect_id(observation),
        "selectionBounds": {"min": min_count, "max": max_count},
        "selectedIndices": selected_indices,
        "candidateCount": len(options),
        "truncated": len(options) > len(candidate_indices),
        "candidates": [
            {
                "index": index,
                "labelJa": _rule_option_label_ja(observation, option),
                "kind": _rule_trace_option_kind(option, _rule_option_id(observation, option)),
                "optionType": _option_type(option),
                "cardId": _rule_option_id(observation, option),
                "targetCardId": _rule_target_id(observation, option),
                "attackId": _attack_id(option),
                "selected": index in selected_indices,
            }
            for index, option in ((candidate_index, options[candidate_index]) for candidate_index in candidate_indices)
        ],
    }


def _rule_decision_trace(observation, options, selected_indices):
    select = _read(observation, "select", {})
    if _select_context(select) in (0, "main"):
        return _rule_main_decision_trace(observation, options, selected_indices)
    return _rule_optional_decision_trace(observation, options, selected_indices)


def _rule_static_trace_payload():
    global RULE_TRACE_STATIC_EMITTED
    include_static_details = not RULE_TRACE_STATIC_EMITTED
    payload = {
        "ruleStaticIncluded": include_static_details,
        "ruleDocs": {
            "flowMarkdown": "competition/ptcg_abc/rule_docs/rule_flow_ja.md",
            "flowHtml": "competition/ptcg_abc/rule_docs/rule_flow_ja.html",
            "flowChartHtml": "competition/ptcg_abc/rule_docs/rule_flow_chart_ja.html",
        },
        "ruleOrderJa": [
            {"order": order, "phase": phase_name, "description": description}
            for order, phase_name, description in RULE_ONLY_PHASES_JA
        ],
    }
    if include_static_details:
        payload["ruleDetailJa"] = [
            {"order": order, "phase": phase_name, "condition": condition, "action": action_text}
            for order, phase_name, condition, action_text in RULE_ONLY_DETAIL_PHASES_JA
        ]
        payload["optionalRuleDetailJa"] = [
            {"order": order, "phase": phase_name, "condition": condition, "action": action_text}
            for order, phase_name, condition, action_text in RULE_ONLY_OPTIONAL_DETAIL_PHASES_JA
        ]
        RULE_TRACE_STATIC_EMITTED = True
    else:
        payload["ruleDetailJa"] = []
        payload["optionalRuleDetailJa"] = []
        payload["ruleStaticOmittedReasonJa"] = "リプレイ軽量化のため、詳細な固定ルール表はこのエージェント実行中の初回トレースだけに含めます。各手の判断根拠は decisionTrace を参照してください。"
    return payload


def _rule_choose_main_action_core(observation, options):
    select = _read(observation, "select", {})
    if _select_context(select) not in (0, "main"):
        return None

    current, _, player = _current_player(observation)
    supporter_played = _supporter_played_this_turn(current, player)

    alakazam_recovery = _rule_find_alakazam_articuno_recovery_option(observation, options)
    if alakazam_recovery is not None:
        return alakazam_recovery

    alakazam_escape = _rule_find_alakazam_porygon_escape_option(observation, options)
    if alakazam_escape is not None:
        return alakazam_escape

    low_deck_finish = _rule_low_deck_finish_action_index(observation, options)
    if low_deck_finish is not None:
        return low_deck_finish

    if supporter_played:
        if _rule_post_support_rocket_feather_damage_shortage(observation, options):
            factory = _rule_find_post_support_factory_option(observation, options)
            if factory is not None:
                return factory
            fuel_search = _rule_find_post_support_rocket_feather_fuel_search_option(observation, options)
            if fuel_search is not None:
                return fuel_search
        pre_factory_attack_condition_search = _rule_find_post_support_pre_factory_attack_condition_search_option(observation, options)
        if pre_factory_attack_condition_search is not None:
            return pre_factory_attack_condition_search
        factory = _rule_find_post_support_factory_option(observation, options)
        if factory is not None:
            return factory
        miracle_high_prize_ko = _rule_find_miracle_headset_high_prize_ko_option(observation, options)
        if miracle_high_prize_ko is not None:
            return miracle_high_prize_ko

    if not supporter_played:
        giovanni_lethal = _rule_find_supporter_option(observation, options, GIOVANNI)
        if giovanni_lethal is not None and _sakaki_can_take_remaining_prizes(observation, giovanni_from_hand=True):
            return giovanni_lethal

    alakazam_basic = _rule_find_alakazam_basic_play_option(observation, options)
    if alakazam_basic is not None:
        return alakazam_basic

    dragapult_basic = _rule_find_dragapult_basic_play_option(observation, options)
    if dragapult_basic is not None:
        return dragapult_basic

    for evolution_id in (HONCHKROW, PORYGON2):
        evolution = _rule_find_evolution_option(observation, options, evolution_id)
        if evolution is not None:
            return evolution

    basic = _rule_find_basic_play_option(observation, options, (MURKROW, PORYGON))
    if basic is not None:
        return basic

    post_support_poke_pad = _rule_find_post_support_poke_pad_option(observation, options)
    if post_support_poke_pad is not None:
        return post_support_poke_pad

    opening_attach = _rule_find_opening_team_rocket_attach_option(observation, options)
    if opening_attach is not None:
        return opening_attach

    if not supporter_played:
        pre_support_attach = _rule_find_pre_support_attach_option(observation, options)
        if pre_support_attach is not None:
            return pre_support_attach

        miracle_high_prize_ko = _rule_find_miracle_headset_high_prize_ko_option(observation, options)
        if miracle_high_prize_ko is not None:
            return miracle_high_prize_ko

        supporter = _rule_find_supporter_to_play(observation, options)
        pre_support_factory = _rule_find_pre_support_factory_before_supporter_option(observation, options, supporter)
        if pre_support_factory is not None:
            return pre_support_factory
        ready_supporter_id = _rule_option_id(observation, options[supporter]) if supporter is not None else None
        if ready_supporter_id == GIOVANNI and _sakaki_can_take_remaining_prizes(observation, giovanni_from_hand=True):
            return supporter
        if ready_supporter_id == PROTON:
            return supporter
        dragapult_articuno_needed = _dragapult_articuno_search_needed(observation)
        skip_support_search_items = ready_supporter_id in (ARIANA, ARCHER, PROTON) and not dragapult_articuno_needed

        draw_search_blocked = _rule_low_deck_draw_search_blocked(observation, options)
        if not draw_search_blocked:
            opening_lance_search = _rule_find_opening_lance_search_item(observation, options)
            if opening_lance_search is not None:
                return opening_lance_search

        if supporter is None and not draw_search_blocked:
            miracle_athena_rescue = _rule_find_miracle_headset_athena_rescue_option(observation, options)
            if miracle_athena_rescue is not None:
                return miracle_athena_rescue

        if not draw_search_blocked:
            for item_id in (POKE_PAD, NIGHT_STRETCHER, TEAM_ROCKET_TRANSCEIVER, POKEGEAR, ROTO_STICK):
                if skip_support_search_items and item_id in (TEAM_ROCKET_TRANSCEIVER, POKEGEAR, ROTO_STICK):
                    continue
                item = _rule_find_card_option(observation, options, item_id, (7, "play"))
                if item is not None:
                    return item

        if supporter is not None:
            return supporter

    factory = _rule_find_post_support_factory_option(observation, options)
    if factory is not None:
        return factory

    miracle_high_prize_ko = _rule_find_miracle_headset_high_prize_ko_option(observation, options)
    if miracle_high_prize_ko is not None:
        return miracle_high_prize_ko

    fuel_search = _rule_find_post_support_rocket_feather_fuel_search_option(observation, options)
    if fuel_search is not None:
        return fuel_search

    attach = _rule_find_attach_option(observation, options)
    if attach is not None:
        return attach

    alakazam_taunt = _rule_find_alakazam_taunt_attack_option(observation, options)
    if alakazam_taunt is not None:
        return alakazam_taunt

    if not _opponent_active_has_hop_dodge_protection(observation):
        attack = _rule_choose_best_main_attack(observation, options)
        if attack is not None:
            return attack

    murkrow_ko_attack = _rule_find_murkrow_ko_attack_option(observation, options)
    if murkrow_ko_attack is not None:
        return murkrow_ko_attack

    hand_supporters = _count_cards(player, ("hand",), lambda card_id: card_id in ROCKET_SUPPORTERS)
    if hand_supporters <= 0 and not _alakazam_lock_strategy_active(observation):
        tempt = _rule_find_attack_option(options, MURKROW_TEMPT_ATTACK)
        if tempt is not None:
            return tempt

    if not _opponent_active_has_hop_dodge_protection(observation):
        tempo_attack = _rule_find_tempo_attack_option(observation, options)
        if tempo_attack is not None:
            return tempo_attack

    for option_index, option in enumerate(options):
        if _option_type(option) in (14, "end"):
            return option_index
    return 0 if options else None


def _rule_choose_main_action(observation, options):
    selected_index = _rule_choose_main_action_core(observation, options)
    return _rule_shallow_search_rerank_main_action(observation, options, selected_index)


def _rule_card_name_ja(card_id):
    if card_id is None:
        return "カードなし"
    return RULE_ONLY_CARD_NAMES_JA.get(card_id, f"カードID {card_id}")


def _rule_attack_name_ja(attack_id):
    if attack_id is None or attack_id == -1:
        return "ワザなし"
    return RULE_ONLY_ATTACK_NAMES_JA.get(attack_id, f"ワザID {attack_id}")


def _rule_action_indices(action):
    if action is None:
        return []
    if isinstance(action, (list, tuple)):
        result = []
        for item in action:
            result.extend(_rule_action_indices(item))
        return result
    if isinstance(action, bool):
        return []
    if isinstance(action, int):
        return [action] if action >= 0 else []
    return []


def _rule_option_label_ja(observation, option):
    option_type = _option_type(option)
    card_id = _rule_option_id(observation, option)
    target_id = _rule_target_id(observation, option)
    attack_id = _attack_id(option)
    if option_type in (7, "play"):
        return f"{_rule_card_name_ja(card_id)}を使う"
    if option_type in (8, "attach"):
        return f"{_rule_card_name_ja(card_id)}を{_rule_card_name_ja(target_id)}につける"
    if option_type in (9, "evolve"):
        return f"{_rule_card_name_ja(target_id)}を{_rule_card_name_ja(card_id)}に進化"
    if option_type in (10, "ability"):
        return f"{_rule_card_name_ja(card_id)}の効果を使う"
    if option_type in (12, "retreat"):
        return "にげる/入れ替え"
    if option_type in (13, "attack"):
        return f"{_rule_attack_name_ja(attack_id)}で攻撃"
    if option_type in (14, "end"):
        return "番を終える"
    if option_type in (3, "card"):
        return f"{_rule_card_name_ja(card_id)}を選ぶ"
    return f"選択肢 type={option_type} / {_rule_card_name_ja(card_id)}"


def _rule_main_phase_ja(observation, options, selected_index, decision_trace=None):
    if selected_index is None or not (0 <= selected_index < len(options)):
        return "フォールバック", "合法な選択肢を最低限返す安全処理。"
    option = options[selected_index]
    option_type = _option_type(option)
    identifier = _rule_option_id(observation, option)
    attack_id = _attack_id(option)
    current, _, player = _current_player(observation)
    supporter_played = _supporter_played_this_turn(current, player)
    stadium_played = _stadium_played_this_turn(current, player)

    if decision_trace and decision_trace.get("type") == "main":
        core_index = decision_trace.get("fixedRuleIndex")
        reranked_index = decision_trace.get("finalIndex")
    else:
        core_index = _rule_choose_main_action_core(observation, options)
        reranked_index = _rule_shallow_search_rerank_main_action(observation, options, core_index)
    if (
        isinstance(core_index, int)
        and isinstance(reranked_index, int)
        and reranked_index != core_index
        and selected_index == reranked_index
    ):
        core_label = _rule_option_label_ja(observation, options[core_index])
        return (
            "軽量先読み補正",
            f"固定ルールの候補は「{core_label}」だったが、上位候補を5〜8個に絞って1〜2手先を軽く評価し、攻撃接続・次ターン到達・探索価値・軽量ニューラル評価の合算が十分に上回ったため置き換えた。攻撃接続しない逃げ、特にエネルギー付きアクティブの逃げは補正側でも大幅減点する。",
        )

    if supporter_played and not stadium_played and identifier == FACTORY:
        if _rule_post_support_rocket_feather_damage_shortage(observation, options):
            return "サポート後ファクトリー", "ロケットフェザーは撃てるが打点燃料が足りないため、先にファクトリーでロケット団サポートを引きにいく。"
        return "サポート後ファクトリー", "サポート使用後に効果解決を挟んで main に戻ったので、攻撃より前にファクトリーを使う。"
    if option_type in (7, "play") and identifier == MIRACLE_HEADSET and _rule_rocket_feather_ko_reachable_after_miracle(observation, min_prize=2):
        return "攻撃前ミラクルインカム", "相手バトル場が2サイド以上で、今の手札燃料だけでは足りないが、トラッシュのロケット団サポートを戻せばロケットフェザーKOに届く。戻す優先度はアテナ、アポロ、ラムダ、サカキ、ランス。"
    if supporter_played and option_type in (7, "play") and identifier == POKE_PAD:
        return "サポート後ポケパッド", "サポート使用後でも、対象があるポケパッドは攻撃や手張りより前に使う。対象がなければ対象なし選択を許容する。"
    if supporter_played and option_type in (7, "play") and identifier in (ROTO_STICK, POKEGEAR, TEAM_ROCKET_TRANSCEIVER):
        if _rule_post_support_rocket_feather_damage_shortage(observation, options):
            return "攻撃前燃料補助", "ロケットフェザーは撃てるが打点が足りないため、ファクトリー後にロトスティック→ポケギア→レシーバーの順でロケット団サポートを探す。"
        if _rule_attack_condition_missing_before_factory(observation, options):
            return "ファクトリー前山札圧縮", "攻撃条件のエネルギーまたは進化先が足りないため、ロトスティック→ポケギア→レシーバーの順で山札のサポートを抜いてからファクトリー2ドローへ進む。"
        return "サポート後サポート探索", "サポート使用後に残ったサポート探索札を、攻撃前の燃料や次ターン選択肢のために使う。"
    if _alakazam_lock_strategy_active(observation) and option_type in (7, "play") and identifier in (ARTICUNO, MURKROW, PORYGON):
        return "対フーディンたね展開", "相手盤面にフーディンLineが見えており、ロケット団のフリーザーが使用可能なので、レジストヴェールでたねのロケット団ポケモンを守る形へ寄せる。フリーザー、必要数のヤミカラス、最低限のポリゴンの順で展開する。"
    if _alakazam_lock_strategy_active(observation) and option_type in (13, "attack") and attack_id in RULE_ONLY_MURKROW_KO_ATTACKS:
        return "対フーディンいちゃもん", "フリーザーのレジストヴェールを維持するため進化せず、ヤミカラスのいちゃもん系ワザだけを使う。フリーザーがサイド落ち確定、または残りサイドを取り切れる数の準備済みアタッカーがそろった時だけ通常攻撃へ戻る。"
    if option_type in (9, "evolve"):
        if _alakazam_resist_veil_plan_active(observation):
            return "対フーディン進化解除", "フリーザーがサイド落ちしていないが、残りサイドを取り切れる数の準備済みヤミカラス/ドンカラスLineがそろったため、進化してサイドレースを終わらせに行く。"
        if _dragapult_matchup_active(observation):
            return "進化: ドラパルト対面例外", "相手盤面にドラパルトLineが見えているため、進化はこのターン攻撃に直結する場合だけ許可する。"
        return "進化", "ドンカラス/ポリゴン2への進化を、サポートや攻撃より前に処理する。"
    if option_type in (7, "play") and identifier in (MURKROW, ARTICUNO, PORYGON):
        return "たね展開", "ベンチを作るため、ヤミカラス/フリーザー/ポリゴンを場へ出す。ドラパルトLineが見えた対面ではフリーザー未設置なら最優先し、設置後にヤミカラス2体、ポリゴンの順で見る。フリーザー未設置で既存ポリゴンLineがある、または残りベンチ1枠なら追加ポリゴンで枠を潰さない。"
    if not supporter_played and option_type in (7, "play") and identifier in (
        POKE_PAD,
        NIGHT_STRETCHER,
        TEAM_ROCKET_TRANSCEIVER,
        POKEGEAR,
        ROTO_STICK,
    ):
        return "サポート前準備", "サポート判断の前に、ポケモンやサポートに触るグッズを使う。"
    if not supporter_played and option_type in (7, "play") and identifier in SUPPORTER_CARD_IDS:
        if identifier == PROTON:
            return "サポート使用: ランス", "ベンチ2体未満、またはヤミカラス/ドンカラス合計2体未満で、ロケット団サポート使用履歴0枚の時だけ使う。ポリゴン系統の有無はランス判断から除外する。"
        if identifier == GIOVANNI:
            return "サポート使用: サカキ", "バトル場の確定KOを基準に、後ろにより高い価値の確定KOがある時だけ使う。後ろに上位KOがなければ前を取る。"
        if identifier == ARIANA:
            return "サポート使用: アテナ", "基本の手札補充/エネルギー探索としてアテナを使う。"
        if identifier == PETREL:
            return "サポート使用: ラムダ", "ファクトリー未設置、ポケパッド経由の進化/盤面形成、またはエネルギー探索へつながる時にラムダを使う。"
        if identifier == ARCHER:
            return "サポート使用: アポロ", "アテナの実ドローが薄い時、手札が重い/少ない時、燃料・進化・エネルギーをまとめて探す時にアポロを使う。"
        return "サポート使用", "そのターンのサポート権を使う。"
    if option_type in (8, "attach"):
        if _alakazam_lock_strategy_active(observation):
            return "対フーディン手張り", "フリーザーのレジストヴェールを維持するため、進化後やフリーザーには寄せず、いちゃもんを使うヤミカラスへロケット団エネルギーを付ける。"
        if identifier == TEAM_ROCKET_ENERGY and _opening_turn_order(current, _safe_int(_read(current, "yourIndex"), 0)) is not None:
            return "初動R団エネルギー", "先攻/後攻1ターン目の指定ルールとして、サポートやグッズより前にロケット団エネルギーの手張りを確認する。"
        if not supporter_played:
            return "サポート前手張り", "サポート使用前に、攻撃準備または後続育成として安全なエネルギーを先に貼って手札を圧縮する。"
        return "エネルギー手張り", "攻撃に必要なポケモン、または後続アタッカーにエネルギーをつける。"
    if option_type in (13, "attack"):
        if attack_id == ROCKET_FEATHER_ATTACK:
            return "攻撃: ロケットフェザー", "必要打点、消費サポート枚数、手札/山札からのサポート補充見込みをRコマンドと比較して選ぶ。コストではアテナ、なければアポロを1枚守るが、KOに必要なら保護札も切る。"
        if attack_id in PORYGON2_R_COMMAND_ATTACKS:
            return "攻撃: Rコマンド", "トラッシュのロケット団サポート枚数による打点を、ロケットフェザーのサポート消費と比較して選ぶ。残りサイド3以下では、KOできるなら手札サポートを消費しない詰め筋として優先する。"
        if attack_id in RULE_ONLY_MURKROW_KO_ATTACKS:
            return "ヤミカラス緊急ワザ: いちゃもん系", "前がヤミカラスで、いちゃもん系ワザで相手バトル場をKOできる時だけ使う。"
        if attack_id == MURKROW_TEMPT_ATTACK:
            return "ヤミカラス緊急ワザ: たぶらかす", "いちゃもん系でKOできず、手札にサポートがない時だけ、たぶらかすを使う。"
        return "攻撃", "他の準備行動がないのでワザを使う。"
    if option_type in (14, "end"):
        return "終了", "使うべきルールが残っていないため番を終える。"
    return "その他", "メインフェーズの残り選択肢を処理する。"


def _rule_optional_phase_ja(observation, options, selected_indices):
    select = _read(observation, "select", {})
    context = _select_context(select)
    if context == 38:
        return "マリガン追加ドロー", "相手のマリガン回数に応じて最大枚数を選ぶ。"
    if context in (4, "promote active pokemon", "context4"):
        return "きぜつ後の前出し", "次ターンの攻撃成立条件を満たしやすいポケモンをバトル場に出す。"
    taunt_lock = _choose_taunt_move_lock_option(observation, options, _selection_bounds(select)[1])
    if taunt_lock is not None:
        return "いちゃもん系ワザロック", "ワザIDだけの候補でも、同じ試合で相手が使ったワザ、単一候補、複数未知なら2個目の順で止める。"
    rocket_feather = _rule_choose_rocket_feather_costs(observation, options, _selection_bounds(select)[1])
    if rocket_feather is not None:
        return "ロケットフェザーのコスト", "KO時も、KOを逃さない範囲でアテナ1枚、なければアポロ1枚を守る。非KOで2ターンKOへ進める時は保護札を残し、残りのロケット団サポートを厚めに捨てる。"
    effect_id = _rule_effect_id(observation)
    min_count, max_count = _selection_bounds(select)
    if effect_id == MIRACLE_HEADSET and _rule_miracle_headset_rocket_feather_fuel_targets(observation, options, max_count):
        return "ミラクルインカムの燃料回収", "2サイド以上の相手バトル場をロケットフェザーで倒すため、不足枚数分をアテナ、アポロ、ラムダ、サカキ、ランスの順で戻す。手札の不要サポートをコストに回す。"
    option_ids = [_rule_option_id(observation, option) for option in options]
    if any(card_id in ROCKET_SUPPORTERS for card_id in option_ids):
        return "サポート選択", "ランス初回、その後はアテナ/ラムダ/アポロ/サカキの優先で選ぶ。"
    if any(card_id in (MURKROW, ARTICUNO, HONCHKROW, PORYGON, PORYGON2) for card_id in option_ids):
        if _alakazam_resist_veil_plan_active(observation):
            if _alakazam_lock_strategy_active(observation):
                return "ポケモン選択: 対フーディン", "フリーザーが使用可能で残りサイド取り切り体制が未完成のため、フリーザー、必要数のヤミカラス、最低限のポリゴンを優先し、攻撃に直結しない進化先は選ばない。"
            return "ポケモン選択: 対フーディン解除", "フリーザー使用不能、または残りサイドを取り切れる準備済みアタッカー数があるため、通常の進化/攻撃へ戻す。"
        if _dragapult_matchup_active(observation):
            return "ポケモン選択: ドラパルト対面", "相手盤面にドラパルトLineが見えているため、フリーザー未設置ならフリーザーを最優先で選ぶ。設置後はヤミカラス2体、ポリゴンの順。フリーザー未設置で既存ポリゴンLineがある、または残りベンチ1枠ならポリゴン選択を止め、攻撃に直結しない進化先も抑制する。"
        return "ポケモン選択", "種切れ回避、ドンカラス進化、ポリゴン2繋ぎの順で選ぶ。"
    if any(card_id in (TEAM_ROCKET_ENERGY, IGNITION_ENERGY) for card_id in option_ids):
        return "エネルギー選択", "ロケット団エネルギーを優先し、必要ならイグニッションを選ぶ。"
    return "任意選択", "その選択画面での優先順に従って選ぶ。"


def abc_lab_explain_action(observation, action):
    try:
        select = _select_payload(observation)
        options = _select_options(select)
        if not options:
            return None
        min_count, max_count = _selection_bounds(select)
        selected = [
            index
            for index in _rule_action_indices(action)
            if 0 <= index < len(options)
        ]
        context = _select_context(select)
        try:
            decision_trace = _rule_decision_trace(observation, options, selected)
        except Exception as trace_exc:
            decision_trace = {
                "type": "trace-error",
                "traceVersion": 2,
                "error": str(trace_exc),
                "selectedIndices": selected,
            }
        if context in (0, "main"):
            chosen = decision_trace.get("finalIndex") if decision_trace.get("type") == "main" else _rule_choose_main_action(observation, options)
            phase, reason = _rule_main_phase_ja(observation, options, selected[0] if selected else chosen, decision_trace)
        else:
            phase, reason = _rule_optional_phase_ja(observation, options, selected)
        selected_options = [
            {
                "index": index,
                "label": _rule_option_label_ja(observation, options[index]),
            }
            for index in selected
        ]
        top_options = [
            {
                "index": index,
                "label": _rule_option_label_ja(observation, option),
            }
            for index, option in enumerate(options[:8])
        ]
        chosen_text = "、".join(item["label"] for item in selected_options) if selected_options else "何も選ばない"
        payload = {
            "schemaVersion": 1,
            "source": "abc-self-play-lab",
            "available": True,
            "engine": "rule-only-ja",
            "context": context,
            "selected": selected,
            "summaryJa": f"【{phase}】{chosen_text}。理由: {reason}",
            "phaseJa": phase,
            "reasonJa": reason,
            "decisionTrace": decision_trace,
            "selectedOptions": selected_options,
            "topOptions": top_options,
            "selectionBounds": {"min": min_count, "max": max_count},
            "codeRefs": [
                {
                    "file": "competition/ptcg_abc/main.py",
                    "symbol": "_rule_choose_main_action",
                    "startLine": None,
                    "endLine": None,
                },
                {
                    "file": "competition/ptcg_abc/main.py",
                    "symbol": "_rule_shallow_search_rerank_main_action",
                    "startLine": None,
                    "endLine": None,
                }
            ],
        }
        payload.update(_rule_static_trace_payload())
        return payload
    except Exception as exc:
        return {
            "schemaVersion": 1,
            "source": "abc-self-play-lab",
            "available": False,
            "engine": "rule-only-ja",
            "summaryJa": f"ルール可視化の生成に失敗しました: {exc}",
            "selected": _rule_action_indices(action),
            "topOptions": [],
            "selectedOptions": [],
            "codeRefs": [],
        }


def _rule_only_agent_impl(obs_dict):
    select = _select_payload(obs_dict)
    if _is_initial_deck_request(obs_dict, select):
        _reset_public_knowledge()
        return MY_DECK

    _remember_public_information(obs_dict)
    _remember_hop_phantump_dodge(obs_dict)
    _remember_observed_attacks(obs_dict)

    options = _select_options(select)
    min_count, max_count = _selection_bounds(select)
    if max_count <= 0 or not options:
        return []

    context = _select_context(select)
    if context not in (0, "main"):
        selected = _rule_choose_optional_multi_select(obs_dict, options, min_count, max_count)
        return _sanitize_action(selected, min_count, max_count, len(options))

    selected_index = _rule_choose_main_action(obs_dict, options)
    if isinstance(selected_index, int):
        return _sanitize_action([selected_index], min_count, max_count, len(options))
    return _sanitize_action([0] if min_count > 0 else [], min_count, max_count, len(options))


def _rule_only_safe_fallback(obs_dict):
    try:
        select = _select_payload(obs_dict)
        if _is_initial_deck_request(obs_dict, select):
            return MY_DECK
        options = _select_options(select)
        min_count, max_count = _selection_bounds(select)
        if max_count <= 0 or not options:
            return []
        context = _select_context(select)
        if context in (0, "main"):
            for option_index, option in enumerate(options):
                if _option_type(option) in (14, "end"):
                    return [option_index]
            for option_index, option in enumerate(options):
                if _option_type(option) in (8, "attach") and _rule_forbidden_energy_attach_option(obs_dict, option):
                    continue
                return _sanitize_action([option_index], min_count, max_count, len(options))
        return _sanitize_action([0] if min_count > 0 else [], min_count, max_count, len(options))
    except Exception:
        return []


def agent(obs_dict):
    try:
        return _rule_only_agent_impl(obs_dict)
    except Exception:
        return _rule_only_safe_fallback(obs_dict)


def select_action(observation, *args, **kwargs):
    return agent(observation)


def get_action(observation, *args, **kwargs):
    return agent(observation)


def act(observation, *args, **kwargs):
    return agent(observation)


def submission_agent(observation, configuration=None):
    # Kaggle's file loader executes the last newly inserted callable, not
    # necessarily the function named "agent". Keep this function last.
    return agent(observation)


# END RULE_ONLY_AGENT
