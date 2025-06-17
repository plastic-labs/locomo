#!/usr/bin/env python3
"""
LOCOMO Dataset Explorer and Subsetter Tool

A comprehensive tool for exploring and subsetting the LOCOMO dataset for long-term memory evaluation.
Supports exploration of conversations, sessions, categories, and intelligent temporal subsetting.
"""

import json
import argparse
import re
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass


@dataclass
class EvidenceRef:
    """Parsed evidence reference (e.g., 'D2:8' -> session=2, message=8)"""
    session: int
    message: int
    
    @classmethod
    def parse(cls, evidence_str: str) -> 'EvidenceRef':
        """Parse evidence string like 'D2:8' into session and message numbers"""
        match = re.match(r'D(\d+):(\d+)', evidence_str)
        if not match:
            raise ValueError(f"Invalid evidence format: {evidence_str}")
        return cls(int(match.group(1)), int(match.group(2)))
    
    def __lt__(self, other: 'EvidenceRef') -> bool:
        """Enable sorting by session first, then message"""
        return (self.session, self.message) < (other.session, other.message)


class LocomoDataset:
    """Efficient loader and analyzer for LOCOMO dataset"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._data = None
        self._conversations_cache = None
    
    @property
    def data(self) -> List[Dict]:
        """Lazy-load dataset"""
        if self._data is None:
            with open(self.file_path, 'r') as f:
                self._data = json.load(f)
        return self._data
    
    def get_conversation_list(self) -> List[Tuple[int, str, str]]:
        """Get list of (index, speaker_a, speaker_b) for all conversations"""
        if self._conversations_cache is None:
            self._conversations_cache = [
                (i, conv['conversation']['speaker_a'], conv['conversation']['speaker_b'])
                for i, conv in enumerate(self.data)
            ]
        return self._conversations_cache
    
    def get_conversation(self, conv_idx: int) -> Dict:
        """Get specific conversation by index"""
        if conv_idx < 0 or conv_idx >= len(self.data):
            raise ValueError(f"Conversation index {conv_idx} out of range (0-{len(self.data)-1})")
        return self.data[conv_idx]
    
    def extract_sessions(self, conversation: Dict) -> Dict[int, Dict]:
        """Extract all sessions from a conversation with metadata"""
        sessions = {}
        conv_data = conversation['conversation']
        
        session_num = 1
        while f'session_{session_num}' in conv_data:
            session_key = f'session_{session_num}'
            datetime_key = f'session_{session_num}_date_time'
            
            sessions[session_num] = {
                'messages': conv_data[session_key],
                'datetime': conv_data.get(datetime_key, 'Unknown'),
                'message_count': len(conv_data[session_key]) if conv_data[session_key] else 0
            }
            session_num += 1
        
        return sessions
    
    def analyze_questions_by_category(self, conversation: Dict) -> Dict[str, Dict]:
        """Analyze question distribution by category and session"""
        questions = conversation['qa']
        
        # Overall category distribution
        category_counts = Counter(q['category'] for q in questions)
        
        # Category distribution by session (based on evidence)
        session_categories = defaultdict(lambda: defaultdict(int))
        
        for question in questions:
            category = question['category']
            evidence_refs = [EvidenceRef.parse(ev) for ev in question['evidence']]
            
            # Count this question for each session it references
            sessions_referenced = set(ref.session for ref in evidence_refs)
            for session in sessions_referenced:
                session_categories[session][category] += 1
        
        return {
            'overall': dict(category_counts),
            'by_session': dict(session_categories)
        }
    
    def find_max_evidence(self, questions: List[Dict]) -> Optional[EvidenceRef]:
        """Find the latest evidence reference across multiple questions"""
        all_evidence = []
        
        for question in questions:
            for evidence_str in question['evidence']:
                all_evidence.append(EvidenceRef.parse(evidence_str))
        
        return max(all_evidence) if all_evidence else None
    
    def create_subset(self, conv_idx: int, category: int, n: int) -> Dict:
        """Create temporal subset based on category and question count"""
        conversation = self.get_conversation(conv_idx)
        
        # Filter questions by category and take first n
        category_questions = [q for q in conversation['qa'] if q['category'] == category]
        selected_questions = category_questions[:n]
        
        if not selected_questions:
            raise ValueError(f"No questions found for category {category}")
        
        # Find latest evidence reference
        max_evidence = self.find_max_evidence(selected_questions)
        if not max_evidence:
            raise ValueError("No evidence found in selected questions")
        
        # Extract sessions up to the max evidence point
        sessions = self.extract_sessions(conversation)
        subset_conversation = {'speaker_a': conversation['conversation']['speaker_a'],
                             'speaker_b': conversation['conversation']['speaker_b']}
        
        # Include all sessions up to and including the target session
        for session_num in range(1, max_evidence.session + 1):
            if session_num in sessions:
                session_data = sessions[session_num]
                subset_conversation[f'session_{session_num}'] = session_data['messages']
                subset_conversation[f'session_{session_num}_date_time'] = session_data['datetime']
                
                # If this is the target session, truncate at the target message
                if session_num == max_evidence.session:
                    messages = session_data['messages']
                    truncated_messages = []
                    
                    for msg in messages:
                        truncated_messages.append(msg)
                        # Stop after including the target message
                        msg_ref = EvidenceRef.parse(msg['dia_id'])
                        if msg_ref.message >= max_evidence.message:
                            break
                    
                    subset_conversation[f'session_{session_num}'] = truncated_messages
        
        return {
            'qa': selected_questions,
            'conversation': subset_conversation,
            'subset_info': {
                'original_conversation': conv_idx,
                'category': category,
                'questions_requested': n,
                'questions_found': len(selected_questions),
                'max_evidence': f"D{max_evidence.session}:{max_evidence.message}",
                'sessions_included': list(range(1, max_evidence.session + 1)),
                'total_messages_included': sum(
                    len(subset_conversation[f'session_{i}']) 
                    for i in range(1, max_evidence.session + 1)
                    if f'session_{i}' in subset_conversation
                )
            }
        }


class LocomoExplorer:
    """Interactive explorer for LOCOMO dataset"""
    
    def __init__(self, dataset: LocomoDataset):
        self.dataset = dataset
    
    def list_conversations(self):
        """Display all available conversations"""
        conversations = self.dataset.get_conversation_list()
        print("Available Conversations:")
        print("=" * 50)
        for idx, speaker_a, speaker_b in conversations:
            print(f"{idx:2d}: {speaker_a} ↔ {speaker_b}")
        print()
    
    def analyze_conversation(self, conv_idx: int):
        """Deep analysis of a specific conversation"""
        conversation = self.dataset.get_conversation(conv_idx)
        conv_list = self.dataset.get_conversation_list()
        speaker_a, speaker_b = conv_list[conv_idx][1], conv_list[conv_idx][2]
        
        print(f"Conversation {conv_idx}: {speaker_a} ↔ {speaker_b}")
        print("=" * 60)
        
        # Session overview
        sessions = self.dataset.extract_sessions(conversation)
        print(f"Sessions: {len(sessions)}")
        for session_num, session_data in sessions.items():
            print(f"  Session {session_num}: {session_data['message_count']} messages ({session_data['datetime']})")
        
        print()
        
        # Question analysis
        print(f"Total Questions: {len(conversation['qa'])}")
        category_analysis = self.dataset.analyze_questions_by_category(conversation)
        
        print("\nQuestions by Category (Overall):")
        for category, count in sorted(category_analysis['overall'].items()):
            print(f"  Category {category}: {count} questions")
        
        print("\nQuestions by Category per Session:")
        for session_num in sorted(category_analysis['by_session'].keys()):
            session_cats = category_analysis['by_session'][session_num]
            cat_str = ", ".join(f"Cat{cat}:{count}" for cat, count in sorted(session_cats.items()))
            print(f"  Session {session_num}: {cat_str}")
        
        # Sample questions per category
        print("\nSample Questions by Category:")
        questions_by_cat = defaultdict(list)
        for q in conversation['qa']:
            questions_by_cat[q['category']].append(q)
        
        for category in sorted(questions_by_cat.keys()):
            sample_q = questions_by_cat[category][0]
            evidence_str = ", ".join(sample_q['evidence'])
            print(f"  Category {category}: \"{sample_q['question']}\" (Evidence: {evidence_str})")
        
        print()
    
    def preview_subset(self, conv_idx: int, category: int, n: int):
        """Preview what a subset would include without creating it"""
        conversation = self.dataset.get_conversation(conv_idx)
        conv_list = self.dataset.get_conversation_list()
        speaker_a, speaker_b = conv_list[conv_idx][1], conv_list[conv_idx][2]
        
        # Get category questions
        category_questions = [q for q in conversation['qa'] if q['category'] == category]
        selected_questions = category_questions[:n]
        
        print(f"Subset Preview: {speaker_a} ↔ {speaker_b} | Category {category} | Top {n} questions")
        print("=" * 80)
        
        if not selected_questions:
            print(f"❌ No questions found for category {category}")
            return
        
        print(f"✓ Found {len(selected_questions)} questions (requested {n})")
        
        # Show selected questions
        print(f"\nSelected Questions:")
        for i, q in enumerate(selected_questions, 1):
            evidence_str = ", ".join(q['evidence'])
            print(f"  {i:2d}. \"{q['question'][:60]}{'...' if len(q['question']) > 60 else ''}\"")
            print(f"      Evidence: {evidence_str}")
        
        # Calculate what would be included
        max_evidence = self.dataset.find_max_evidence(selected_questions)
        if max_evidence:
            print(f"\nLatest Evidence: D{max_evidence.session}:{max_evidence.message}")
            
            sessions = self.dataset.extract_sessions(conversation)
            print(f"Sessions to include: 1 to {max_evidence.session}")
            
            total_messages = 0
            for session_num in range(1, max_evidence.session + 1):
                if session_num in sessions:
                    if session_num < max_evidence.session:
                        count = sessions[session_num]['message_count']
                        total_messages += count
                        print(f"  Session {session_num}: All {count} messages")
                    else:
                        # Calculate partial session
                        count = max_evidence.message
                        total_messages += count
                        print(f"  Session {session_num}: First {count} messages")
            
            print(f"\nTotal messages in subset: {total_messages}")
        print()


def main():
    parser = argparse.ArgumentParser(description="LOCOMO Dataset Explorer and Subsetter")
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Explore mode
    explore_parser = subparsers.add_parser('explore', help='Explore dataset structure')
    explore_parser.add_argument('--list-conversations', action='store_true',
                               help='List all available conversations')
    explore_parser.add_argument('--conversation', type=int, metavar='N',
                               help='Analyze specific conversation (0-9)')
    explore_parser.add_argument('--category', type=int, metavar='N',
                               help='Category for preview (use with --preview)')
    explore_parser.add_argument('--n', type=int, default=10, metavar='N',
                               help='Number of questions (default: 10)')
    explore_parser.add_argument('--preview', action='store_true',
                               help='Preview subset (requires --conversation, --category)')
    
    # Subset mode
    subset_parser = subparsers.add_parser('subset', help='Create dataset subset')
    subset_parser.add_argument('--conversation', type=int, required=True, metavar='N',
                              help='Conversation index (0-9)')
    subset_parser.add_argument('--category', type=int, required=True, metavar='N',
                              help='Category to filter by')
    subset_parser.add_argument('--n', type=int, default=10, metavar='N',
                              help='Number of questions to include (default: 10)')
    subset_parser.add_argument('--output', type=str, required=True, metavar='FILE',
                              help='Output file for subset')
    
    # Common arguments
    for subparser in [explore_parser, subset_parser]:
        subparser.add_argument('--data', type=str, default='data/locomo10.json',
                              help='Path to LOCOMO dataset (default: data/locomo10.json)')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return
    
    # Load dataset
    try:
        dataset = LocomoDataset(args.data)
        explorer = LocomoExplorer(dataset)
    except FileNotFoundError:
        print(f"❌ Error: Dataset file '{args.data}' not found")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in dataset file: {e}")
        return
    
    # Execute commands
    try:
        if args.mode == 'explore':
            if args.list_conversations:
                explorer.list_conversations()
            
            if args.conversation is not None:
                if args.preview and args.category is not None:
                    explorer.preview_subset(args.conversation, args.category, args.n)
                else:
                    explorer.analyze_conversation(args.conversation)
        
        elif args.mode == 'subset':
            subset_data = dataset.create_subset(args.conversation, args.category, args.n)
            
            with open(args.output, 'w') as f:
                json.dump(subset_data, f, indent=2)
            
            info = subset_data['subset_info']
            print(f"✓ Subset created: {args.output}")
            print(f"  Questions: {info['questions_found']}/{info['questions_requested']}")
            print(f"  Max evidence: {info['max_evidence']}")
            print(f"  Sessions: {len(info['sessions_included'])}")
            print(f"  Total messages: {info['total_messages_included']}")
    
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == '__main__':
    main() 