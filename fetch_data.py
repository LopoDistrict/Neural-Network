import xml.etree.ElementTree as ET
import requests
from bs4 import BeautifulSoup
import time
import json
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re
from urllib.parse import urljoin
import random
from pathlib import Path
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QAPair:
    """Data structure for question-answer pairs"""
    post_id: str
    question: str
    answer: str
    metadata: Dict[str, any]

class StackExchangeScraper:
    """
    Scraper for Stack Exchange data using XML IDs and web scraping
    Designed to work with your neural network training pipeline
    """
    
    def __init__(self, base_url: str = "https://stackoverflow.com/questions/", 
                 delay_range: Tuple[float, float] = (1.0, 3.0)):
        """
        Initialize scraper
        
        Args:
            base_url: Base URL for the Stack Exchange site
            delay_range: Min/max delay between requests (seconds)
        """
        self.base_url = base_url
        self.delay_range = delay_range
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.qa_pairs = []
        self.failed_ids = []
        
    def extract_ids_from_xml(self, xml_file_path: str) -> List[str]:
        """
        Extract Post IDs from XML file
        
        Args:
            xml_file_path: Path to your XML file
            
        Returns:
            List of post IDs
        """
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            
            post_ids = []
            
            # Extract PostTypeId from each row
            for row in root.findall('.//row'):
                post_id = row.get('PostTypeId')
                if post_id:
                    post_ids.append(post_id)
            
            # Remove duplicates while preserving order
            unique_ids = list(dict.fromkeys(post_ids))
            
            logger.info(f"Extracted {len(unique_ids)} unique Post Type IDs from {xml_file_path}")
            return unique_ids
            
        except Exception as e:
            logger.error(f"Error parsing XML file {xml_file_path}: {str(e)}")
            return []
    
    def scrape_question_answer(self, post_id: str) -> Optional[QAPair]:
        """
        Scrape question and answer from a single post ID
        
        Args:
            post_id: Stack Exchange post ID
            
        Returns:
            QAPair object or None if scraping failed
        """
        url = f"{self.base_url}{post_id}"
        
        try:
            # Add random delay to be respectful
            delay = random.uniform(*self.delay_range)
            time.sleep(delay)
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract question
            question_elem = soup.find('div', {'class': 's-prose js-post-body'})
            if not question_elem:
                question_elem = soup.find('div', {'class': 'post-text'})
            
            if not question_elem:
                logger.warning(f"No question found for post {post_id}")
                return None
            
            question_title = soup.find('h1', {'class': 'fs-headline1'})
            if not question_title:
                question_title = soup.find('a', {'class': 'question-hyperlink'})
            
            question_text = self.clean_text(question_elem.get_text())
            title_text = self.clean_text(question_title.get_text()) if question_title else ""
            
            # Combine title and question
            full_question = f"{title_text}\n{question_text}".strip() if title_text else question_text
            
            # Extract best answer (first answer with accepted class or highest voted)
            answer_elem = soup.find('div', {'class': 'accepted-answer'})
            if not answer_elem:
                # Find first answer
                answer_elem = soup.find('div', {'class': 'answer'})
            
            if not answer_elem:
                logger.warning(f"No answer found for post {post_id}")
                return None
            
            answer_text_elem = answer_elem.find('div', {'class': 's-prose js-post-body'})
            if not answer_text_elem:
                answer_text_elem = answer_elem.find('div', {'class': 'post-text'})
            
            if not answer_text_elem:
                logger.warning(f"No answer text found for post {post_id}")
                return None
            
            answer_text = self.clean_text(answer_text_elem.get_text())
            
            # Validate question and answer quality
            if len(full_question.strip()) < 10 or len(answer_text.strip()) < 10:
                logger.warning(f"Question or answer too short for post {post_id}")
                return None
            
            # Extract metadata
            metadata = {
                'post_id': post_id,
                'url': url,
                'scraped_at': time.time(),
                'question_length': len(full_question),
                'answer_length': len(answer_text)
            }
            
            # Try to extract tags
            tags_elem = soup.find_all('a', {'class': 'post-tag'})
            if tags_elem:
                metadata['tags'] = [tag.get_text().strip() for tag in tags_elem[:5]]
            
            qa_pair = QAPair(
                post_id=post_id,
                question=full_question,
                answer=answer_text,
                metadata=metadata
            )
            
            logger.info(f"✅ Successfully scraped post {post_id}")
            return qa_pair
            
        except Exception as e:
            logger.error(f"❌ Error scraping post {post_id}: {str(e)}")
            self.failed_ids.append(post_id)
            return None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize scraped text"""
        if not text:
            return ""
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove code block markers if any
        text = re.sub(r'```[\s\S]*?```', '[CODE_BLOCK]', text)
        text = re.sub(r'`[^`]+`', '[CODE]', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Limit length to reasonable size
        if len(text) > 2000:
            text = text[:2000] + "..."
        
        return text.strip()
    
    def batch_scrape(self, post_ids: List[str], 
                    max_items: Optional[int] = None,
                    save_progress_every: int = 50) -> List[QAPair]:
        """
        Scrape multiple posts with progress saving
        
        Args:
            post_ids: List of post IDs to scrape
            max_items: Maximum number of items to scrape (None for all)
            save_progress_every: Save progress every N items
            
        Returns:
            List of QAPair objects
        """
        if max_items:
            post_ids = post_ids[:max_items]
        
        logger.info(f"Starting batch scrape of {len(post_ids)} posts")
        
        scraped_count = 0
        failed_count = 0
        
        for i, post_id in enumerate(post_ids):
            try:
                qa_pair = self.scrape_question_answer(post_id)
                
                if qa_pair:
                    self.qa_pairs.append(qa_pair)
                    scraped_count += 1
                else:
                    failed_count += 1
                
                # Progress update
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i + 1}/{len(post_ids)} - Success: {scraped_count}, Failed: {failed_count}")
                
                # Save progress periodically
                if (i + 1) % save_progress_every == 0:
                    self.save_progress(f"progress_checkpoint_{i + 1}.pkl")
                
            except KeyboardInterrupt:
                logger.info("Scraping interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error processing post {post_id}: {str(e)}")
                failed_count += 1
        
        logger.info(f"Batch scrape completed: {scraped_count} successful, {failed_count} failed")
        return self.qa_pairs
    
    def save_progress(self, filename: str):
        """Save current progress to file"""
        data = {
            'qa_pairs': self.qa_pairs,
            'failed_ids': self.failed_ids,
            'timestamp': time.time()
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Progress saved to {filename}")
    
    def load_progress(self, filename: str):
        """Load progress from file"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            self.qa_pairs = data['qa_pairs']
            self.failed_ids = data['failed_ids']
            
            logger.info(f"Progress loaded from {filename}: {len(self.qa_pairs)} Q&A pairs")
            return True
            
        except Exception as e:
            logger.error(f"Error loading progress: {str(e)}")
            return False
    
    def to_training_format(self, format_style: str = "chat") -> List[str]:
        """
        Convert scraped Q&A pairs to training format for your neural network
        
        Args:
            format_style: "chat", "instruction", or "simple"
            
        Returns:
            List of formatted training strings
        """
        training_data = []
        
        for qa in self.qa_pairs:
            if format_style == "chat":
                formatted = f"<Human>: {qa.question}\n<Assistant>: {qa.answer}"
            elif format_style == "instruction":
                formatted = f"### Question:\n{qa.question}\n\n### Answer:\n{qa.answer}"
            elif format_style == "simple":
                formatted = f"Q: {qa.question}\nA: {qa.answer}"
            else:
                formatted = f"{qa.question}\n{qa.answer}"
            
            training_data.append(formatted)
        
        return training_data
    
    def save_as_json(self, filename: str):
        """Save Q&A pairs as JSON"""
        data = []
        for qa in self.qa_pairs:
            data.append({
                'post_id': qa.post_id,
                'question': qa.question,
                'answer': qa.answer,
                'metadata': qa.metadata
            })
        
        with open(filename, 'a', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Data saved as JSON to {filename}")
    
    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about scraped data"""
        if not self.qa_pairs:
            return {}
        
        question_lengths = [len(qa.question.split()) for qa in self.qa_pairs]
        answer_lengths = [len(qa.answer.split()) for qa in self.qa_pairs]
        
        return {
            'total_pairs': len(self.qa_pairs),
            'failed_scrapes': len(self.failed_ids),
            'avg_question_length': sum(question_lengths) / len(question_lengths),
            'avg_answer_length': sum(answer_lengths) / len(answer_lengths),
            'max_question_length': max(question_lengths),
            'max_answer_length': max(answer_lengths),
            'min_question_length': min(question_lengths),
            'min_answer_length': min(answer_lengths)
        }

# Integration function for your neural network
def process_xml_and_scrape_for_training(xml_file_path: str,
                                      base_url: str = "https://stackoverflow.com/questions/",
                                      max_items: int = 100,
                                      output_file: str = "scraped_qa_data.json") -> List[str]:
    """
    Complete pipeline: XML -> IDs -> Scraping -> Training Data
    
    Args:
        xml_file_path: Path to your XML file
        base_url: Base URL for scraping
        max_items: Maximum number of items to scrape
        output_file: File to save scraped data
        
    Returns:
        List of training strings ready for your neural network
    """
    
    logger.info("🚀 Starting XML processing and scraping pipeline")
    
    # Initialize scraper
    scraper = StackExchangeScraper(base_url=base_url)
    
    # Extract IDs from XML
    logger.info("📋 Extracting Post IDs from XML...")
    post_ids = scraper.extract_ids_from_xml(xml_file_path)
    
    if not post_ids:
        logger.error("No Post IDs found in XML file")
        return []
    
    logger.info(f"Found {len(post_ids)} Post IDs")
    
    # Scrape Q&A pairs
    logger.info("🕷️ Starting web scraping...")
    qa_pairs = scraper.batch_scrape(post_ids, max_items=max_items)
    
    if not qa_pairs:
        logger.error("No Q&A pairs successfully scraped")
        return []
    
    # Save scraped data
    scraper.save_as_json(output_file)
    
    # Convert to training format
    training_data = scraper.to_training_format("chat")
    
    # Print statistics
    stats = scraper.get_statistics()
    logger.info("📊 Scraping Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info(f"✅ Pipeline completed! Generated {len(training_data)} training examples")
    
    return training_data

# Quick test function
def quick_test_scraper():
    """Test the scraper with a few sample IDs"""
    scraper = StackExchangeScraper()
    
    # Test with a few known Stack Overflow post IDs
    test_ids = ["2", "11", "1732348"]  # Famous Stack Overflow posts
    
    logger.info("🧪 Testing scraper with sample posts...")
    
    for post_id in test_ids:
        qa_pair = scraper.scrape_question_answer(post_id)
        if qa_pair:
            print(f"\n✅ Post {post_id}:")
            print(f"Question: {qa_pair.question[:100]}...")
            print(f"Answer: {qa_pair.answer[:100]}...")
        else:
            print(f"❌ Failed to scrape post {post_id}")


def initialise_scraper(filename: str, url: str, max_items: int):
    # Get user input
    xml_file = filename #input("Enter path to your XML file: ").strip()
    
    if not xml_file or not Path(xml_file).exists():
        print("❌ XML file not found. Running quick test instead...")
        quick_test_scraper()
    else:
        base_url = "https://" + url #input("Enter base URL (default: https://stackoverflow.com/questions/): ").strip()
        if not base_url:
            base_url = "https://stackoverflow.com/questions/"
        
        max_items = input("Maximum items to scrape (default: 50): ").strip()
        max_items = int(max_items) if max_items.isdigit() else 50
        
        # Run the complete pipeline
        training_data = process_xml_and_scrape_for_training(
            xml_file_path=xml_file,
            base_url=base_url,
            max_items=max_items
        )
        
        if training_data:
            print(f"\n🎉 Success! Generated {len(training_data)} training examples")
            print("\nFirst example:")
            print(training_data[0][:500] + "..." if len(training_data[0]) > 500 else training_data[0])
            
            # Option to save as text file for easy inspection
            #save_txt = input("\nSave training data as text file? (y/n): ").lower().strip()
            #if save_txt == 'y':
            with open("training_data.txt", "a", encoding="utf-8") as f:
                for i, example in enumerate(training_data):
                    f.write(f"=== Example {i+1} ===\n{example}\n\n")
            print("💾 Training data saved to training_data.txt")


if __name__ == "__main__":
    print("🕷️ Stack Exchange Scraper for Neural Network Training")
    print("=" * 60)
    
    print("This script will scrape Stack Exchange data for training the neural network.")
    
    print("getting scrapped data from Stack Exchange...")
    list_url = ("stackoverflow.com/questions/", "superuser.com/questions/", "serverfault.com/questions/")
    list_file = ("data.xml", "data2.xml", "serverfault.com")
    for filename, url in list_file, list_url:
        initialise_scraper(filename, url, max_items=50) #Initialise scrapper at 50 items
    print("🎉🎉🎉 Data scraping completed. Check the output files for results.")


    