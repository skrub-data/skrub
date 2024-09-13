describe('test sorting the summary stats columns', () => {
    it('sorts the table when clicking arrows', () => {
        cy.get('@report').find('[data-test="summary-statistics-tab"]')
            .click();
        cy.get('@report').find('.summary-stats-table').as('table');
        cy.get('@table').find('tbody tr').first().should('have.attr',
            'data-column-name', 'gender');
        cy.get('@report').contains('Column name').as('colName');
        cy.get('@colName').parent().find('button').first().as('colNameButton').click();
        cy.get('@colNameButton').should('have.attr', 'data-is-active');
        cy.get('@table').find('tbody tr').first().should('have.attr',
                                                         'data-column-name', 'assignment_category');
        cy.get('@report').find('th').contains('Unique values').as(
            'unique');
        cy.get('@unique').parent().find('button').first().as('uniqueButton').click();
        cy.get('@uniqueButton').should('have.attr', 'data-is-active');
        cy.get('@colNameButton').should('not.have.attr', 'data-is-active');
        cy.get('@table').find('tbody tr').first().should('have.attr',
            'data-column-name', 'gender');
        cy.get('@table').find('tbody tr').last().should('have.attr',
            'data-column-name', 'year_first_hired');
        cy.get('@unique').parent().find('button').first().next()
    .click();
        cy.get('@table').find('tbody tr').first().should('have.attr',
            'data-column-name', 'date_first_hired');
        cy.get('@table').find('tbody tr').last().should('have.attr',
            'data-column-name', 'year_first_hired');
    });
});
