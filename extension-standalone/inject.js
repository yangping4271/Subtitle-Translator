// This script runs in the MAIN world (page context), so it can access custom element properties like .visibility
window.addEventListener('YTSP_OpenTranscript', () => {
    const engagementPanel = document.querySelector('ytd-engagement-panel-section-list-renderer[target-id="engagement-panel-searchable-transcript"]');
    if (engagementPanel) {
        // Force the panel to open by setting the Polymer property
        engagementPanel.visibility = "ENGAGEMENT_PANEL_VISIBILITY_EXPANDED";
    } else {
        console.warn('YouTube SubtitlePlus: Engagement panel not found in Main World.');
    }
});
