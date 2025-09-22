// Entity type mappings for styling
const entityStyleMap = {
    // Basic entities
    'PERSON': 'entity-person',
    'ORG': 'entity-organization',
    'GPE': 'entity-location',
    'LOC': 'entity-location',
    'DATE': 'entity-date',
    'TIME': 'entity-time',
    'MONEY': 'entity-money',
    'QUANTITY': 'entity-quantity',
    'CARDINAL': 'entity-quantity',
    'ORDINAL': 'entity-quantity',
    'PERCENT': 'entity-quantity',
    
    // Contact information
    'EMAIL': 'entity-email',
    'PHONE': 'entity-phone',
    
    // Technical
    'URL': 'entity-url',
    'IP_ADDRESS': 'entity-ip_address',
    
    // Financial
    'CREDIT_CARD': 'entity-credit_card',
    'BITCOIN': 'entity-bitcoin',
    'SSN': 'entity-ssn',
    'ROUTING_NUMBER': 'entity-routing_number',
    
    // Suspicious
    'SUSPICIOUS_FILE': 'entity-suspicious_file'
};

const categoryLabels = {
    'PEOPLE': 'People',
    'ORGANIZATIONS': 'Organizations',
    'LOCATIONS': 'Locations',
    'DATES': 'Dates',
    'TIMES': 'Times',
    'MONEY': 'Money',
    'QUANTITIES': 'Quantities',
    'CONTACT_INFO': 'Contact Information',
    'TECHNICAL': 'Technical',
    'FINANCIAL': 'Financial',
    'SUSPICIOUS': 'Suspicious',
    'OTHER': 'Other'
};

async function analyzeText() {
    const textInput = document.getElementById('textInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = document.getElementById('btnText');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('resultsSection');
    
    const text = textInput.value.trim();
    
    if (!text) {
        alert('Please enter some text for analysis!');
        return;
    }
    
    // Show loading state
    analyzeBtn.disabled = true;
    btnText.style.display = 'none';
    loader.style.display = 'inline';
    
    try {
        // Call the Flask API
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const data = await response.json();
        
        // Display highlighted text only
        displayHighlightedText(text, data.entities);
        showResultsSection();
        
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred during analysis. Please try again.');
    } finally {
        // Reset button state
        analyzeBtn.disabled = false;
        btnText.style.display = 'inline';
        loader.style.display = 'none';
    }
}

function showResultsSection() {
    const inputSection = document.getElementById('inputSection');
    const resultsSection = document.getElementById('resultsSection');
    const legendSection = document.getElementById('legendSection');
    
    console.log('Showing results section...');
    console.log('Legend element found:', legendSection !== null);
    
    inputSection.style.display = 'none';
    legendSection.style.display = 'block'; // Keep legend visible
    resultsSection.style.display = 'block';
    
    console.log('Legend display style:', legendSection.style.display);
}

function showInputSection() {
    const inputSection = document.getElementById('inputSection');
    const resultsSection = document.getElementById('resultsSection');
    const legendSection = document.getElementById('legendSection');
    
    inputSection.style.display = 'block';
    legendSection.style.display = 'block';
    resultsSection.style.display = 'none';
}

function displayHighlightedText(text, entities) {
    const highlightedTextDiv = document.getElementById('highlightedText');
    
    // Sort entities by start position (ascending) for proper processing
    const sortedEntities = entities.sort((a, b) => a.start - b.start);
    
    let result = '';
    let lastIndex = 0;
    
    // Process each entity in order
    sortedEntities.forEach(entity => {
        // Add text before this entity
        if (entity.start > lastIndex) {
            result += text.substring(lastIndex, entity.start);
        }
        
        // Add the highlighted entity
        const styleClass = entityStyleMap[entity.label] || 'entity-other';
        const tagLabel = getTagLabel(entity.label);
        result += `<span class="${styleClass}">${entity.text}</span><span class="entity-tag">${tagLabel}</span>`;
        
        lastIndex = entity.end;
    });
    
    // Add any remaining text after the last entity
    if (lastIndex < text.length) {
        result += text.substring(lastIndex);
    }
    
    highlightedTextDiv.innerHTML = result;
}

function getTagLabel(label) {
    const tagMap = {
        'PERSON': 'PERSON',
        'ORG': 'ORGANIZATION',
        'GPE': 'LOCATION',
        'LOC': 'LOCATION',
        'DATE': 'DATE',
        'TIME': 'TIME',
        'MONEY': 'MONEY',
        'QUANTITY': 'QUANTITY',
        'CARDINAL': 'QUANTITY',
        'ORDINAL': 'QUANTITY',
        'PERCENT': 'QUANTITY',
        'EMAIL': 'EMAIL',
        'PHONE': 'PHONE',
        'URL': 'URL',
        'IP_ADDRESS': 'IP',
        'CREDIT_CARD': 'CARD',
        'BITCOIN': 'BITCOIN',
        'SSN': 'SSN',
        'ROUTING_NUMBER': 'ROUTING',
        'SUSPICIOUS_FILE': 'SUSPICIOUS_FILE'
    };
    return tagMap[label] || 'OTHER';
}


// Allow Enter key to trigger analysis (Ctrl+Enter)
document.getElementById('textInput').addEventListener('keydown', function(event) {
    if (event.ctrlKey && event.key === 'Enter') {
        analyzeText();
    }
});

// Add some sample text on page load for demonstration
window.addEventListener('load', function() {
    const sampleText = `URGENT: Your account has been suspended!

Hello Sarah Johnson,

This is URGENT! Please verify your account immediately at https://suspicious-bank.tk
Your credit card 1234-5678-9012-3456 will expire soon and needs immediate attention.

Contact us at support@fake-bank.com or call +1-555-123-4567
Download this important file: account_verification.exe

Bitcoin payment: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
SSN: 123-45-6789

Meeting with Microsoft on September 20th, 2025 at 3:00 PM in New York.
Contract value: $50,000

Best regards,
Mike Anderson`;
    
    document.getElementById('textInput').value = sampleText;
});
