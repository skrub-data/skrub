describe('test the dataframe sample tab', () => {
    it('shows a column card when clicking a column and can close it', () => {
        cy.get('@report').find(
            '[data-test="click-on-table-announcement"]').as(
            'announcement').should('be.visible');
        cy.get('@report').find('#sample-table-bar-display').as('bar')
            .should('not.be.visible');
        cy.get('@bar').should('have.text', '');

        cy.get('@report').find('td[data-column-idx="1"]').first()
            .click();
        cy.get('@announcement').should('not.be.visible');
        cy.get('@bar').should('be.visible');
        cy.get('@bar').should('have.text', 'POL');
        cy.get('@report').find('[data-test="table-bar-select"]').as(
            'barSelect').select('columnNameRepr');
        cy.get('@bar').should('have.text', "'department'");
        cy.get('@report').find('#col_1_in_sample_tab').as('col1Card')
            .should('be.visible');

        cy.get('@report').find('td[data-column-idx="2"]').first()
            .click();
        cy.get('@bar').should('have.text', "'department_name'");
        cy.get('@report').find('#col_1_in_sample_tab').should(
            'not.be.visible');
        cy.get('@report').find('#col_2_in_sample_tab').as('col2Card')
            .should('be.visible');

        cy.get('@col2Card').find('.close-card-button').click();
        cy.get('@col2Card').should('not.be.visible');
        cy.get('@announcement').should('be.visible');
        cy.get('@bar').should('have.text', "");
        cy.get('@bar').should('not.be.visible');

        // clicking on the header selects the first cell in the column
        cy.get('@report').find('th[data-column-idx="1"]').first()
            .click();
        cy.get('@report').find('td[data-column-idx="1"]').first()
            .should('have.data', 'isActive', '');
        cy.get('@bar').should('have.text', "'department'");

        cy.get('@report').find('[data-test="column-filter-select"]').as(
                "filter")
            .select('String columns');
        cy.get('@bar').should('be.visible');
        cy.get('@announcement').should('not.be.visible');
        cy.get('@bar').should('have.text', "'department'");

        cy.get('@filter').select('Numeric columns');
        cy.get('@bar').should('not.be.visible');
        cy.get('@announcement').should('be.visible');
        cy.get('@bar').should('have.text', "");
    });

    it('copies the cell contents', () => {
        cy.get('@report').find('td[data-column-idx="3"]').first().click();
        cy.get('body').type('{ctrl}c');
        cy.window().its('navigator.clipboard')
            .then((clip) => clip.readText())
            .should('equal',
                "'MSB Information Mgmt and Tech Division Records Management Section'\n"
                );
    });
});
