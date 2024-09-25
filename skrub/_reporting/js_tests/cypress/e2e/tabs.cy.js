describe('test tabbed interface navigation', () => {
    it('can select tab panel with mouse or keyboard', () => {
        cy.get('@report').find("[data-test='sample-tab']").as(
            'sampleTab').should('have.data', 'isSelected');
        cy.get('@report').find("[data-test='summary-statistics-tab']")
            .as(
                'statsTab').should('not.have.data', 'isSelected');
        cy.get('@report').find("[data-test='summaries-tab']").as(
            'summariesTab').should('not.have.data', 'isSelected');
        cy.get('@report').find("[data-test='associations-tab']").as(
            'associationsTab').should('not.have.data', 'isSelected');

        cy.get('@report').find("[data-test='sample-panel']").as(
            'samplePanel').should('be.visible');
        cy.get('@report').find("[data-test='summary-statistics-panel']")
            .as(
                'statsPanel').should('not.be.visible');
        cy.get('@report').find("[data-test='summaries-panel']").as(
            'summariesPanel').should('not.be.visible');
        cy.get('@report').find("[data-test='associations-panel']").as(
            'associationsPanel').should('not.be.visible');

        cy.get('@summariesTab').click();
        cy.get('@samplePanel').should('not.be.visible');
        cy.get('@summariesPanel').should('be.visible');

        cy.get('@summariesTab').should('have.focus');
        // with modifier does nothing
        cy.get('@summariesTab').type('{shift+rightArrow}');
        cy.get('@summariesPanel').should('be.visible');
        cy.get('@summariesTab').type('{rightArrow}');
        cy.get('@summariesPanel').should('not.be.visible');
        cy.get('@associationsPanel').should('be.visible');

        cy.get('@associationsTab').should('have.focus');
        cy.get('@associationsTab').type('{rightArrow}');
        cy.get('@associationsPanel').should('not.be.visible');
        cy.get('@samplePanel').should('be.visible');

        cy.get('@sampleTab').should('have.focus');
        cy.get('@sampleTab').type('{rightArrow}');
        cy.get('@samplePanel').should('not.be.visible');
        cy.get('@statsPanel').should('be.visible');

        cy.get('@statsTab').should('have.focus');
        cy.get('@statsTab').type('{leftArrow}');
        cy.get('@statsPanel').should('not.be.visible');
        cy.get('@samplePanel').should('be.visible');
    });
});
