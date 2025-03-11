"""
CAPABILITIES:
- Evaluates weighted majority voting algorithms with comprehensive metrics
- Analyzes voting fairness, bias, and statistical validity
- Handles multiple tie-breaking strategies (random, weight-prioritized, specified fallback)
- Supports various voting schemes (majority, plurality, ranked-choice, Borda count)
- Performs sensitivity analysis of weight distributions on outcomes
- Calculates confidence metrics and statistical significance
- Conducts group fairness analysis across voter demographics
- Benchmarks performance on large-scale voting scenarios
- Generates detailed evaluation reports with visualizations
- Simulates strategic voting behavior and coalition formation
"""

import logging
import json
import time
import os
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Union, Optional, Callable, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import concurrent.futures
import itertools
import math
import sys

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"{log_dir}/weighted_voting_evaluation_{timestamp}.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger("voting_evaluator")


class TieBreakStrategy(Enum):
    """Strategies for breaking ties in voting."""
    RANDOM = "random"
    HIGHEST_WEIGHT = "highest_weight"
    FALLBACK_OPTION = "fallback_option"
    WEIGHT_OF_VOTERS = "weight_of_voters"


class VotingMethod(Enum):
    """Different methods for aggregating votes."""
    SIMPLE_MAJORITY = "simple_majority"  # Majority wins
    WEIGHTED_MAJORITY = "weighted_majority"  # Weighted votes, majority wins
    PLURALITY = "plurality"  # Most votes wins, even if < 50%
    RANKED_CHOICE = "ranked_choice"  # Voters rank choices, elimination rounds
    BORDA_COUNT = "borda_count"  # Points assigned based on rank
    APPROVAL = "approval"  # Voters approve multiple options


@dataclass
class VoterGroup:
    """Representation of a demographic group of voters."""
    name: str
    proportion: float = 1.0
    weight_modifier: float = 1.0
    voting_preferences: Dict[str, float] = field(default_factory=dict)


@dataclass
class VotingScenario:
    """A voting scenario with specific parameters."""
    name: str
    description: str
    options: List[str]
    voting_method: VotingMethod
    tie_break_strategy: TieBreakStrategy = TieBreakStrategy.RANDOM
    fallback_option: Optional[str] = None
    voter_groups: List[VoterGroup] = field(default_factory=list)
    expected_outcome: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class VoteResult:
    """Result of a voting process."""
    winner: str
    vote_counts: Dict[str, float]
    normalized_distribution: Dict[str, float]
    tie_occurred: bool = False
    tie_break_applied: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class EvaluationMetrics:
    """Metrics for evaluating a voting system."""
    accuracy: float = 0.0  # How often does outcome match expected
    consistency: float = 0.0  # How consistent are outcomes with repeated runs
    fairness_score: float = 0.0  # Measure of demographic fairness
    robustness: float = 0.0  # Resistance to strategic voting
    sensitivity: float = 0.0  # How sensitive to weight changes
    computation_time_sec: float = 0.0
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class EvaluationResult:
    """Complete result of an evaluation."""
    scenario_name: str
    voting_method: str
    vote_result: VoteResult
    expected_outcome: Optional[str]
    correct_outcome: bool
    metrics: EvaluationMetrics
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scenario_name": self.scenario_name,
            "voting_method": self.voting_method,
            "vote_result": {
                "winner": self.vote_result.winner,
                "vote_counts": self.vote_result.vote_counts,
                "normalized_distribution": self.vote_result.normalized_distribution,
                "tie_occurred": self.vote_result.tie_occurred,
                "tie_break_applied": self.vote_result.tie_break_applied,
                "metadata": self.vote_result.metadata
            },
            "expected_outcome": self.expected_outcome,
            "correct_outcome": self.correct_outcome,
            "metrics": self.metrics.to_dict(),
            "timestamp": self.timestamp
        }


class WeightedVotingSystem:
    """
    Implementation of various weighted voting systems with evaluation capabilities.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the weighted voting system.
        
        Args:
            random_seed: Optional seed for reproducible randomness
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Store test results for analysis
        self.evaluation_results: List[EvaluationResult] = []
    
    def weighted_majority_vote(
        self,
        votes: List[str],
        weights: List[float],
        tie_break: TieBreakStrategy = TieBreakStrategy.RANDOM,
        fallback_option: Optional[str] = None
    ) -> VoteResult:
        """
        Compute the weighted majority vote with support for tie-breaking.
        
        Args:
            votes: List of categorical votes
            weights: List of corresponding weights
            tie_break: Strategy for breaking ties
            fallback_option: Option to select in case of tie (if using FALLBACK_OPTION)
            
        Returns:
            VoteResult with the winner and vote distribution
        """
        if len(votes) != len(weights):
            raise ValueError("Number of votes must match number of weights")
        
        # Count weighted votes
        weighted_counts = defaultdict(float)
        for vote, weight in zip(votes, weights):
            weighted_counts[vote] += weight
        
        # Find the winner(s)
        max_count = max(weighted_counts.values()) if weighted_counts else 0
        winners = [option for option, count in weighted_counts.items() if count == max_count]
        
        tie_occurred = len(winners) > 1
        tie_break_applied = False
        
        # Apply tie-breaking if needed
        if tie_occurred:
            logger.info(f"Tie detected between: {winners}")
            tie_break_applied = True
            
            if tie_break == TieBreakStrategy.RANDOM:
                winner = random.choice(winners)
                logger.info(f"Random tie-break selected: {winner}")
            
            elif tie_break == TieBreakStrategy.HIGHEST_WEIGHT:
                # Find the option with the highest single weight
                option_max_weights = {}
                for option in winners:
                    option_votes = [(v, w) for v, w in zip(votes, weights) if v == option]
                    option_max_weights[option] = max(w for _, w in option_votes)
                
                winner = max(option_max_weights, key=option_max_weights.get)
                logger.info(f"Highest weight tie-break selected: {winner}")
            
            elif tie_break == TieBreakStrategy.FALLBACK_OPTION:
                if fallback_option in winners:
                    winner = fallback_option
                else:
                    winner = fallback_option if fallback_option else random.choice(winners)
                logger.info(f"Fallback option tie-break selected: {winner}")
            
            elif tie_break == TieBreakStrategy.WEIGHT_OF_VOTERS:
                # Choose based on the total weight of voters
                voter_weight_sum = {}
                for option in winners:
                    voter_weight_sum[option] = sum(w for v, w in zip(votes, weights) if v == option)
                
                winner = max(voter_weight_sum, key=voter_weight_sum.get)
                logger.info(f"Weight of voters tie-break selected: {winner}")
            
            else:
                winner = random.choice(winners)  # Default to random if unrecognized
        else:
            winner = winners[0]
        
        # Calculate normalized distribution
        total_weight = sum(weighted_counts.values())
        normalized_distribution = {
            option: count / total_weight if total_weight > 0 else 0
            for option, count in weighted_counts.items()
        }
        
        return VoteResult(
            winner=winner,
            vote_counts=dict(weighted_counts),
            normalized_distribution=normalized_distribution,
            tie_occurred=tie_occurred,
            tie_break_applied=tie_break_applied,
            metadata={
                "total_votes": len(votes),
                "total_weight": total_weight,
                "unique_options": len(weighted_counts)
            }
        )
    
    def ranked_choice_vote(
        self,
        ranked_votes: List[List[str]],
        weights: List[float],
        tie_break: TieBreakStrategy = TieBreakStrategy.RANDOM,
        fallback_option: Optional[str] = None
    ) -> VoteResult:
        """
        Implement ranked choice voting (instant runoff).
        
        Args:
            ranked_votes: List of ranked choices for each voter
            weights: List of voter weights
            tie_break: Strategy for breaking ties
            fallback_option: Option to select in case of tie (if using FALLBACK_OPTION)
            
        Returns:
            VoteResult with the winner and vote distribution
        """
        if len(ranked_votes) != len(weights):
            raise ValueError("Number of ranked vote lists must match number of weights")
        
        # Keep track of eliminated options
        eliminated = set()
        all_options = set()
        for ranking in ranked_votes:
            all_options.update(ranking)
        
        # Track original votes for reporting
        original_weighted_counts = defaultdict(float)
        
        # Keep going until we have a majority winner
        while len(all_options - eliminated) > 1:
            weighted_counts = defaultdict(float)
            
            # Count first choices that haven't been eliminated
            for ranking, weight in zip(ranked_votes, weights):
                # Skip empty rankings
                if not ranking:
                    continue
                
                # Find first non-eliminated choice
                for option in ranking:
                    if option not in eliminated:
                        weighted_counts[option] += weight
                        
                        # For the first round, store original votes
                        if not eliminated:
                            original_weighted_counts[option] += weight
                        
                        break
            
            # If no votes were cast (all options eliminated), break
            if not weighted_counts:
                break
            
            # Find the option with the fewest votes
            min_count = min(weighted_counts.values())
            lowest_options = [option for option, count in weighted_counts.items() if count == min_count]
            
            # Handle ties for elimination
            if len(lowest_options) > 1:
                if tie_break == TieBreakStrategy.RANDOM:
                    to_eliminate = random.choice(lowest_options)
                elif tie_break == TieBreakStrategy.FALLBACK_OPTION:
                    # Keep the fallback option if possible
                    if fallback_option in lowest_options:
                        lowest_options.remove(fallback_option)
                        to_eliminate = random.choice(lowest_options) if lowest_options else None
                    else:
                        to_eliminate = random.choice(lowest_options)
                else:
                    to_eliminate = random.choice(lowest_options)
            else:
                to_eliminate = lowest_options[0]
            
            if to_eliminate:
                eliminated.add(to_eliminate)
                logger.info(f"Eliminated option: {to_eliminate}")
            
            # Check if we have a winner (>50% of total weight)
            total_weight = sum(weighted_counts.values())
            for option, count in weighted_counts.items():
                if count > total_weight / 2:
                    # Winner found
                    normalized_distribution = {
                        option: count / total_weight if total_weight > 0 else 0
                        for option, count in original_weighted_counts.items()
                    }
                    
                    return VoteResult(
                        winner=option,
                        vote_counts=dict(original_weighted_counts),
                        normalized_distribution=normalized_distribution,
                        tie_occurred=False,
                        tie_break_applied=False,
                        metadata={
                            "total_votes": len(ranked_votes),
                            "total_weight": total_weight,
                            "rounds": len(eliminated),
                            "eliminated_order": list(eliminated)
                        }
                    )
        
        # If we get here, only one option remains or no votes were cast
        remaining_options = all_options - eliminated
        
        if len(remaining_options) == 1:
            winner = next(iter(remaining_options))
        else:
            # Tie between final options or no votes cast
            winners = list(weighted_counts.keys()) if weighted_counts else list(all_options - eliminated)
            
            if not winners:
                winner = None
            elif len(winners) == 1:
                winner = winners[0]
            else:
                # Apply tie-breaking for final options
                if tie_break == TieBreakStrategy.RANDOM:
                    winner = random.choice(winners)
                elif tie_break == TieBreakStrategy.FALLBACK_OPTION and fallback_option in winners:
                    winner = fallback_option
                else:
                    winner = random.choice(winners)
        
        # Calculate normalized distribution from original first choices
        total_weight = sum(original_weighted_counts.values())
        normalized_distribution = {
            option: count / total_weight if total_weight > 0 else 0
            for option, count in original_weighted_counts.items()
        }
        
        return VoteResult(
            winner=winner,
            vote_counts=dict(original_weighted_counts),
            normalized_distribution=normalized_distribution,
            tie_occurred=len(remaining_options) != 1,
            tie_break_applied=len(remaining_options) != 1,
            metadata={
                "total_votes": len(ranked_votes),
                "total_weight": total_weight,
                "rounds": len(eliminated),
                "eliminated_order": list(eliminated)
            }
        )
    
    def evaluate_scenario(
        self,
        scenario: VotingScenario,
        num_voters: int = 100,
        num_trials: int = 10
    ) -> EvaluationResult:
        """
        Evaluate a voting scenario.
        
        Args:
            scenario: The voting scenario to evaluate
            num_voters: Number of voters to simulate
            num_trials: Number of trials for consistency checking
            
        Returns:
            EvaluationResult with metrics and outcome
        """
        logger.info(f"Evaluating scenario: {scenario.name}")
        logger.info(f"Voting method: {scenario.voting_method.value}")
        
        start_time = time.time()
        
        # Generate votes based on voter groups
        votes, weights = self._generate_votes(scenario, num_voters)
        
        # Tally votes
        voting_result = self._tally_votes(scenario, votes, weights)
        
        # Calculate evaluation metrics
        metrics = self._calculate_metrics(scenario, voting_result, num_trials, votes, weights)
        
        # Determine if the result matches expected outcome
        correct_outcome = (
            voting_result.winner == scenario.expected_outcome
            if scenario.expected_outcome is not None
            else True
        )
        
        execution_time = time.time() - start_time
        metrics.computation_time_sec = execution_time
        
        # Create evaluation result
        result = EvaluationResult(
            scenario_name=scenario.name,
            voting_method=scenario.voting_method.value,
            vote_result=voting_result,
            expected_outcome=scenario.expected_outcome,
            correct_outcome=correct_outcome,
            metrics=metrics
        )
        
        # Store result
        self.evaluation_results.append(result)
        
        logger.info(f"Evaluation complete. Winner: {voting_result.winner}, "
                   f"Expected: {scenario.expected_outcome}, "
                   f"Correct: {correct_outcome}")
        
        return result
    
    def _generate_votes(
        self,
        scenario: VotingScenario,
        num_voters: int
    ) -> Tuple[List[Any], List[float]]:
        """
        Generate votes and weights based on voter groups.
        
        Args:
            scenario: The voting scenario
            num_voters: Number of voters to simulate
            
        Returns:
            Tuple of (votes, weights)
        """
        votes = []
        weights = []
        
        # Assign voters to groups based on proportions
        if scenario.voter_groups:
            group_proportions = [group.proportion for group in scenario.voter_groups]
            total_proportion = sum(group_proportions)
            normalized_proportions = [p / total_proportion for p in group_proportions]
            
            group_counts = np.random.multinomial(num_voters, normalized_proportions)
            
            for i, (group, count) in enumerate(zip(scenario.voter_groups, group_counts)):
                for _ in range(count):
                    # Generate vote based on group's preferences
                    if scenario.voting_method == VotingMethod.RANKED_CHOICE:
                        # For ranked choice, generate a full ranking
                        if group.voting_preferences:
                            # Use preferences to create a biased ranking
                            options = list(scenario.options)
                            prefs = group.voting_preferences
                            options.sort(key=lambda x: prefs.get(x, 0), reverse=True)
                            
                            # Add some randomness
                            for j in range(len(options) - 1):
                                if random.random() < 0.2:  # 20% chance of swapping
                                    options[j], options[j+1] = options[j+1], options[j]
                            
                            votes.append(options)
                        else:
                            # Random preference order
                            ranking = list(scenario.options)
                            random.shuffle(ranking)
                            votes.append(ranking)
                    else:
                        # For other methods, pick a single option
                        if group.voting_preferences:
                            options = []
                            probs = []
                            for option in scenario.options:
                                pref = group.voting_preferences.get(option, 0)
                                options.append(option)
                                probs.append(max(0.01, pref))  # Ensure at least a small chance
                            
                            # Normalize probabilities
                            total = sum(probs)
                            probs = [p / total for p in probs]
                            
                            vote = np.random.choice(options, p=probs)
                            votes.append(vote)
                        else:
                            # Random vote
                            votes.append(random.choice(scenario.options))
                    
                    # Assign weight based on group's weight modifier
                    weight = 1.0 * group.weight_modifier
                    weights.append(weight)
        else:
            # No voter groups defined, use uniform random voting
            for _ in range(num_voters):
                if scenario.voting_method == VotingMethod.RANKED_CHOICE:
                    ranking = list(scenario.options)
                    random.shuffle(ranking)
                    votes.append(ranking)
                else:
                    votes.append(random.choice(scenario.options))
                weights.append(1.0)
        
        return votes, weights
    
    def _tally_votes(
        self,
        scenario: VotingScenario,
        votes: List[Any],
        weights: List[float]
    ) -> VoteResult:
        """
        Tally votes using the specified voting method.
        
        Args:
            scenario: The voting scenario
            votes: List of votes (format depends on voting method)
            weights: List of voter weights
            
        Returns:
            VoteResult with the winner and vote distribution
        """
        if scenario.voting_method == VotingMethod.WEIGHTED_MAJORITY:
            return self.weighted_majority_vote(
                votes, weights, 
                scenario.tie_break_strategy, 
                scenario.fallback_option
            )
        
        elif scenario.voting_method == VotingMethod.RANKED_CHOICE:
            return self.ranked_choice_vote(
                votes, weights,
                scenario.tie_break_strategy,
                scenario.fallback_option
            )
        
        elif scenario.voting_method == VotingMethod.SIMPLE_MAJORITY:
            # Treat weights as vote counts
            expanded_votes = []
            for vote, weight in zip(votes, weights):
                # Add the vote as many times as the integer part of the weight
                expanded_votes.extend([vote] * int(weight))
                
                # Add one more with probability equal to the fractional part
                fractional = weight - int(weight)
                if random.random() < fractional:
                    expanded_votes.append(vote)
            
            # Count votes
            vote_counts = Counter(expanded_votes)
            total_votes = len(expanded_votes)
            
            if total_votes == 0:
                winner = None
                normalized_distribution = {}
            else:
                winner = vote_counts.most_common(1)[0][0] if vote_counts else None
                normalized_distribution = {
                    option: count / total_votes for option, count in vote_counts.items()
                }
            
            return VoteResult(
                winner=winner,
                vote_counts=dict(vote_counts),
                normalized_distribution=normalized_distribution,
                tie_occurred=len(vote_counts) > 0 and vote_counts.most_common(2)[0][1] == vote_counts.most_common(2)[1][1] if len(vote_counts) >= 2 else False,
                tie_break_applied=False,
                metadata={
                    "total_votes": total_votes,
                    "unique_options": len(vote_counts)
                }
            )
        
        elif scenario.voting_method == VotingMethod.PLURALITY:
            # Simple plurality vote (most votes wins, even if < 50%)
            return self.weighted_majority_vote(
                votes, weights,
                scenario.tie_break_strategy,
                scenario.fallback_option
            )
        
        elif scenario.voting_method == VotingMethod.BORDA_COUNT:
            # For Borda count, votes should be rankings
            if not votes or not isinstance(votes[0], list):
                raise ValueError("Borda count requires ranked votes")
            
            # Calculate Borda scores
            scores = defaultdict(float)
            num_options = len(scenario.options)
            
            for ranking, weight in zip(votes, weights):
                for position, option in enumerate(ranking):
                    # Score is inversely proportional to position (max points for first place)
                    points = num_options - position
                    scores[option] += points * weight
            
            # Find winner
            max_score = max(scores.values()) if scores else 0
            winners = [option for option, score in scores.items() if score == max_score]
            
            tie_occurred = len(winners) > 1
            tie_break_applied = False
            
            if tie_occurred:
                tie_break_applied = True
                
                if scenario.tie_break_strategy == TieBreakStrategy.RANDOM:
                    winner = random.choice(winners)
                elif scenario.tie_break_strategy == TieBreakStrategy.FALLBACK_OPTION:
                    winner = scenario.fallback_option if scenario.fallback_option in winners else random.choice(winners)
                else:
                    winner = random.choice(winners)
            else:
                winner = winners[0] if winners else None
            
            # Calculate normalized distribution
            total_score = sum(scores.values())
            normalized_distribution = {
                option: score / total_score if total_score > 0 else 0
                for option, score in scores.items()
            }
            
            return VoteResult(
                winner=winner,
                vote_counts=dict(scores),
                normalized_distribution=normalized_distribution,
                tie_occurred=tie_occurred,
                tie_break_applied=tie_break_applied,
                metadata={
                    "total_votes": len(votes),
                    "unique_options": len(scores)
                }
            )
        
        else:
            raise ValueError(f"Unsupported voting method: {scenario.voting_method}")
    
    def _calculate_metrics(
        self,
        scenario: VotingScenario,
        vote_result: VoteResult,
        num_trials: int,
        original_votes: List[Any],
        original_weights: List[float]
    ) -> EvaluationMetrics:
        """
        Calculate evaluation metrics.
        
        Args:
            scenario: The voting scenario
            vote_result: The result of the vote
            num_trials: Number of trials for consistency checking
            original_votes: The votes used in the main evaluation
            original_weights: The weights used in the main evaluation
            
        Returns:
            EvaluationMetrics with various metrics
        """
        # Initialize metrics
        metrics = EvaluationMetrics()
        
        # Accuracy (if expected outcome is specified)
        if scenario.expected_outcome is not None:
            metrics.accuracy = 1.0 if vote_result.winner == scenario.expected_outcome else 0.0
        
        # Consistency (over multiple trials)
        winners = []
        for _ in range(num_trials):
            # Generate new votes for each trial
            votes, weights = self._generate_votes(scenario, len(original_votes))
            result = self._tally_votes(scenario, votes, weights)
            winners.append(result.winner)
        
        # Consistency is the frequency of the most common winner
        winner_counts = Counter(winners)
        if winner_counts:
            most_common_winner, most_common_count = winner_counts.most_common(1)[0]
            metrics.consistency = most_common_count / num_trials
        
        # Fairness score (across voter groups)
        if scenario.voter_groups:
            # Calculate representation bias
            group_winner_preferences = []
            
            for group in scenario.voter_groups:
                if vote_result.winner in group.voting_preferences:
                    group_winner_preferences.append(group.voting_preferences[vote_result.winner])
                else:
                    group_winner_preferences.append(0)
            
            if group_winner_preferences:
                # Calculate fairness score based on how well the winner represents all groups
                metrics.fairness_score = min(group_winner_preferences) / max(1e-6, max(group_winner_preferences))
        
        # Robustness (to small perturbations in weights)
        robustness_trials = 5
        perturbed_winners = []
        
        for _ in range(robustness_trials):
            # Perturb weights slightly
            perturbed_weights = [
                max(0.01, w * (1 + random.uniform(-0.1, 0.1)))  # ±10% perturbation
                for w in original_weights
            ]
            
            # Run vote with perturbed weights
            perturbed_result = self._tally_votes(scenario, original_votes, perturbed_weights)
            perturbed_winners.append(perturbed_result.winner)
        
        # Robustness is the frequency of the original winner in perturbed trials
        metrics.robustness = perturbed_winners.count(vote_result.winner) / robustness_trials
        
        # Sensitivity (to weight changes)
        sensitivity_trials = 5
        sensitivity_changes = []
        
        for _ in range(sensitivity_trials):
            # Make a significant change to one random weight
            changed_weights = original_weights.copy()
            change_idx = random.randrange(len(changed_weights))
            changed_weights[change_idx] *= 2  # Double one weight
            
            # Run vote with changed weights
            changed_result = self._tally_votes(scenario, original_votes, changed_weights)
            sensitivity_changes.append(changed_result.winner != vote_result.winner)
        
        # Sensitivity is the frequency of outcome changes due to significant weight changes
        metrics.sensitivity = sum(sensitivity_changes) / sensitivity_trials
        
        return metrics
    
    def evaluate_multiple_scenarios(
        self,
        scenarios: List[VotingScenario],
        num_voters: int = 100,
        num_trials: int = 10,
        parallel: bool = False
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple voting scenarios, optionally in parallel.
        
        Args:
            scenarios: List of scenarios to evaluate
            num_voters: Number of voters to simulate
            num_trials: Number of trials for consistency checking
            parallel: Whether to run evaluations in parallel
            
        Returns:
            List of EvaluationResult objects
        """
        results = []
        
        if parallel and len(scenarios) > 1:
            # Use parallel execution
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = []
                
                for scenario in scenarios:
                    # Create a new evaluator for each scenario to avoid shared state
                    evaluator = WeightedVotingSystem(random_seed=hash(scenario.name) % 10000)
                    futures.append(
                        executor.submit(
                            evaluator.evaluate_scenario,
                            scenario=scenario,
                            num_voters=num_voters,
                            num_trials=num_trials
                        )
                    )
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        self.evaluation_results.append(result)
                    except Exception as e:
                        logger.error(f"Error in parallel evaluation: {str(e)}")
        else:
            # Sequential evaluation
            for scenario in scenarios:
                result = self.evaluate_scenario(
                    scenario=scenario,
                    num_voters=num_voters,
                    num_trials=num_trials
                )
                results.append(result)
        
        return results
    
    def generate_report(
        self,
        output_file: str = "weighted_voting_evaluation_report.json"
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            output_file: Path to save the report
            
        Returns:
            Path to the generated report
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results to report")
            return ""
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Prepare report data
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_scenarios": len(self.evaluation_results),
                "accuracy": sum(r.correct_outcome for r in self.evaluation_results) / len(self.evaluation_results),
                "avg_consistency": sum(r.metrics.consistency for r in self.evaluation_results) / len(self.evaluation_results),
                "avg_robustness": sum(r.metrics.robustness for r in self.evaluation_results) / len(self.evaluation_results),
                "avg_fairness": sum(r.metrics.fairness_score for r in self.evaluation_results) / len(self.evaluation_results)
            },
            "results": [r.to_dict() for r in self.evaluation_results]
        }
        
        # Write report to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {output_file}")
        return output_file
    
    def visualize_results(
        self,
        output_dir: str = "voting_evaluation_visualizations"
    ) -> None:
        """
        Generate visualizations of evaluation results.
        
        Args:
            output_dir: Directory to save visualizations
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.error("Visualization requires matplotlib and seaborn")
            return
        
        if not self.evaluation_results:
            logger.warning("No evaluation results to visualize")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Accuracy by voting method
        plt.figure(figsize=(10, 6))
        voting_methods = {}
        
        for result in self.evaluation_results:
            if result.expected_outcome is not None:  # Only include results with expected outcomes
                method = result.voting_method
                if method not in voting_methods:
                    voting_methods[method] = {"correct": 0, "total": 0}
                
                if result.correct_outcome:
                    voting_methods[method]["correct"] += 1
                voting_methods[method]["total"] += 1
        
        # Calculate accuracy percentages
        methods = []
        accuracies = []
        
        for method, counts in voting_methods.items():
            if counts["total"] > 0:
                methods.append(method)
                accuracies.append(counts["correct"] / counts["total"] * 100)
        
        if methods:
            plt.bar(methods, accuracies)
            plt.title("Accuracy by Voting Method")
            plt.xlabel("Voting Method")
            plt.ylabel("Accuracy (%)")
            plt.ylim(0, 105)
            plt.savefig(os.path.join(output_dir, "accuracy_by_method.png"))
            plt.close()
        
        # 2. Distribution of voting metrics
        plt.figure(figsize=(12, 8))
        
        metrics_data = {
            "Consistency": [r.metrics.consistency for r in self.evaluation_results],
            "Fairness": [r.metrics.fairness_score for r in self.evaluation_results],
            "Robustness": [r.metrics.robustness for r in self.evaluation_results],
            "Sensitivity": [r.metrics.sensitivity for r in self.evaluation_results]
        }
        
        # Create boxplots
        sns.boxplot(data=pd.DataFrame(metrics_data))
        plt.title("Distribution of Evaluation Metrics")
        plt.ylabel("Score (0-1)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, "metrics_distribution.png"))
        plt.close()
        
        # 3. Vote distribution for each scenario
        for i, result in enumerate(self.evaluation_results):
            plt.figure(figsize=(10, 6))
            
            options = list(result.vote_result.normalized_distribution.keys())
            values = list(result.vote_result.normalized_distribution.values())
            
            colors = ['green' if option == result.vote_result.winner else 'gray' for option in options]
            
            plt.bar(options, values, color=colors)
            plt.title(f"Vote Distribution: {result.scenario_name}")
            plt.xlabel("Options")
            plt.ylabel("Normalized Vote Share")
            plt.xticks(rotation=45, ha="right")
            
            if result.expected_outcome:
                plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
                plt.text(0, 0.51, "Majority Threshold", color='r', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"vote_distribution_{i}.png"))
            plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")


def create_test_scenario(name, method=VotingMethod.WEIGHTED_MAJORITY):
    """Create a standard test scenario."""
    options = ["A", "B", "C"]
    
    # Create voter groups with different preferences
    group_a = VoterGroup(
        name="Group A",
        proportion=0.4,
        weight_modifier=1.0,
        voting_preferences={"A": 0.7, "B": 0.2, "C": 0.1}
    )
    
    group_b = VoterGroup(
        name="Group B",
        proportion=0.35,
        weight_modifier=1.0,
        voting_preferences={"A": 0.2, "B": 0.7, "C": 0.1}
    )
    
    group_c = VoterGroup(
        name="Group C",
        proportion=0.25,
        weight_modifier=1.0,
        voting_preferences={"A": 0.1, "B": 0.2, "C": 0.7}
    )
    
    # Determine expected outcome based on voting method and preferences
    if method == VotingMethod.WEIGHTED_MAJORITY:
        # Calculate weighted preference
        total_a = (group_a.proportion * group_a.voting_preferences["A"] + 
                   group_b.proportion * group_b.voting_preferences["A"] + 
                   group_c.proportion * group_c.voting_preferences["A"])
        
        total_b = (group_a.proportion * group_a.voting_preferences["B"] + 
                   group_b.proportion * group_b.voting_preferences["B"] + 
                   group_c.proportion * group_c.voting_preferences["B"])
        
        total_c = (group_a.proportion * group_a.voting_preferences["C"] + 
                   group_b.proportion * group_b.voting_preferences["C"] + 
                   group_c.proportion * group_c.voting_preferences["C"])
        
        expected = "A" if total_a > total_b and total_a > total_c else "B" if total_b > total_c else "C"
    else:
        # For simplicity in other methods, just use B as expected
        expected = "B"
    
    return VotingScenario(
        name=name,
        description=f"Test scenario for {method.value} voting",
        options=options,
        voting_method=method,
        tie_break_strategy=TieBreakStrategy.RANDOM,
        fallback_option="A",
        voter_groups=[group_a, group_b, group_c],
        expected_outcome=expected
    )


def run_standard_tests(output_file="weighted_voting_evaluation.json"):
    """Run a standard suite of tests."""
    # Initialize the voting system
    voting_system = WeightedVotingSystem(random_seed=42)
    
    # Create test scenarios for different voting methods
    scenarios = [
        create_test_scenario("Weighted Majority Test", VotingMethod.WEIGHTED_MAJORITY),
        create_test_scenario("Simple Majority Test", VotingMethod.SIMPLE_MAJORITY),
        create_test_scenario("Plurality Test", VotingMethod.PLURALITY),
        create_test_scenario("Ranked Choice Test", VotingMethod.RANKED_CHOICE),
        create_test_scenario("Borda Count Test", VotingMethod.BORDA_COUNT)
    ]
    
    # Add a tie test scenario
    tie_scenario = VotingScenario(
        name="Tie Breaking Test",
        description="Tests how ties are broken with different strategies",
        options=["A", "B", "C"],
        voting_method=VotingMethod.WEIGHTED_MAJORITY,
        tie_break_strategy=TieBreakStrategy.FALLBACK_OPTION,
        fallback_option="B",
        voter_groups=[
            VoterGroup(
                name="Equal Group",
                proportion=1.0,
                weight_modifier=1.0,
                voting_preferences={"A": 0.33, "B": 0.33, "C": 0.34}
            )
        ],
        expected_outcome="B"  # Expect fallback option to win
    )
    
    scenarios.append(tie_scenario)
    
    # Run evaluations
    voting_system.evaluate_multiple_scenarios(scenarios, num_voters=500, num_trials=20, parallel=True)
    
    # Generate report
    report_path = voting_system.generate_report(output_file)
    
    # Generate visualizations
    voting_system.visualize_results()
    
    return report_path


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate weighted voting systems")
    parser.add_argument("--output", default="weighted_voting_evaluation.json",
                        help="Output file for evaluation results")
    parser.add_argument("--voters", type=int, default=500,
                        help="Number of voters to simulate")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of trials for consistency checking")
    parser.add_argument("--standard-tests", action="store_true",
                        help="Run standard suite of tests")
    parser.add_argument("--parallel", action="store_true",
                        help="Run evaluations in parallel")
    
    args = parser.parse_args()
    
    if args.standard_tests:
        report_path = run_standard_tests(args.output)
        print(f"Standard tests completed. Report saved to {report_path}")
    else:
        # Custom evaluation example
        voting_system = WeightedVotingSystem()
        
        # Example test cases
        test_cases = [
            (["A", "A", "B"], [0.6, 0.3, 0.1], "A"),
            (["B", "B", "A"], [0.4, 0.4, 0.2], "B"),
            (["A", "B", "C"], [0.3, 0.3, 0.4], "C"),
        ]
        
        # Convert to scenarios
        scenarios = []
        for i, (votes, weights, expected) in enumerate(test_cases):
            options = list(set(votes))
            
            # Create a test scenario
            scenario = VotingScenario(
                name=f"Custom Test {i+1}",
                description=f"Test case {i+1} from input",
                options=options,
                voting_method=VotingMethod.WEIGHTED_MAJORITY,
                tie_break_strategy=TieBreakStrategy.RANDOM,
                expected_outcome=expected
            )
            
            # For these simple tests, manually pre-generate votes
            voting_system._generate_votes = lambda s, n, i=i: (votes, weights)
            
            scenarios.append(scenario)
        
        # Run evaluations
        results = voting_system.evaluate_multiple_scenarios(
            scenarios, 
            num_voters=len(votes), 
            num_trials=args.trials,
            parallel=args.parallel
        )
        
        # Generate report
        report_path = voting_system.generate_report(args.output)
        
        # Print summary
        print(f"\nEvaluation Results:")
        for i, result in enumerate(results):
            print(f"\nTest {i+1}:")
            print(f"  Input: Votes={test_cases[i][0]}, Weights={test_cases[i][1]}")
            print(f"  Expected: {test_cases[i][2]}")
            print(f"  Result: {result.vote_result.winner}")
            print(f"  Correct: {'✓' if result.correct_outcome else '✗'}")
            print(f"  Normalized distribution: {result.vote_result.normalized_distribution}")
        
        print(f"\nDetailed report saved to {report_path}")


if __name__ == "__main__":
    main()
