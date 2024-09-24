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
        cy.get('@report').find('#col_1_in_sample_tab').as('col1Card')
            .should('be.visible');

        cy.get('@report').find('td[data-column-idx="2"]').first()
            .click();
        cy.get('@bar').should('have.text', "Department of Police");
        cy.get('@report').find('#col_1_in_sample_tab').should(
            'not.be.visible');
        cy.get('@report').find('#col_2_in_sample_tab').as('col2Card')
            .should('be.visible');

        cy.get('@col2Card').find('.close-card-button').click();
        cy.get('@col2Card').should('not.be.visible');
        cy.get('@announcement').should('be.visible');
        cy.get('@bar').should('have.text', "");
        cy.get('@bar').should('not.be.visible');

        cy.get('@report').find('th[data-column-idx="1"]').first().as(
                'header')
            .click();
        cy.get('@header').should('have.data', 'isActive', '');
        cy.get('@bar').should('have.text', "department");

        cy.get('@report').find('[data-test="column-filter-select"]').as(
                "filter")
            .select('String columns');
        cy.get('@bar').should('be.visible');
        cy.get('@announcement').should('not.be.visible');
        cy.get('@bar').should('have.text', "department");

        cy.get('@filter').select('Numeric columns');
        cy.get('@bar').should('not.be.visible');
        cy.get('@announcement').should('be.visible');
        cy.get('@bar').should('have.text', "");
    });

    it('can navigate in the table with the arrow keys', () => {
        cy.visit('_reports/mini.html');
        cy.get('skrub-table-report').shadow().as('report');

        // hide the numeric columns
        cy.get('@report').find('[data-test="column-filter-select"]').as(
                "filter")
            .select('String columns');

        // type arrow down on the first header should move to first cell
        cy.get('@report').find('th').contains('c 1').as('c1').click();
        cy.get('@c1').should('have.focus');
        cy.get('@c1').type('{downArrow}');
        cy.get('@report').find('td').contains('v 00').as('v00').should(
            'have.focus');

        // move down once more
        cy.get('@v00').type('{downArrow}');
        cy.get('@report').find('td').contains('v 01').as('v01').should(
            'have.focus');
        // with a modifier key does nothing
        cy.get('@v00').type('{shift+downArrow}');
        cy.get('@v00').should('have.focus');

        cy.get('@v01').type('{leftArrow}');
        cy.get('@report').find('th[data-role="index-level-value"]')
            .contains('1').as('index1').should('have.focus');
        cy.get('@index1').should('have.focus');

        // we are on the leftmost visible column so move left should do nothing
        cy.get('@index1').type('{leftArrow}');
        cy.get('@index1').should('have.focus');

        // move right should go to the next visible column, skipping over the
        // hidden numeric one
        cy.get('@index1').type('{rightArrow}');
        cy.get('@v01').type('{rightArrow}');
        cy.get('@report').find('td').contains('v 11').as('v11').should(
            'have.focus');

        // move focus to just above the dataframe head/tail split
        cy.get('@report').find('td').contains('v 14').as('v14').click();

        // move down should skip over the split into the tail
        cy.get('@v14').type('{downArrow}');
        cy.get('@report').find('td').contains('v 15').as('v15').should(
            'have.focus');

        // on the last row move down should not do anything
        cy.get('@v15').type('{downArrow}');
        cy.get('@report').find('td').contains('v 16').as('v16').should(
            'have.focus');
        cy.get('@v16').type('{downArrow}');
        cy.get('@v16').should('have.focus');

        // the card corresponding to the cell we navigated to should be visible
        cy.get('@report').find('.card-header').contains('c 3').as(
            'c3card').should('be.visible');

        // escape deselects and closes the card
        cy.get('@v16').type('{esc}');
        cy.get('@v16').should('not.have.focus');
        cy.get('@c3card').should('not.be.visible');
    });

    it('works with multi-index and multi-columns', () => {
        cy.visit('_reports/multi_index.html');
        cy.get('skrub-table-report').shadow().as('report');
        cy.get('@report').find('#sample-table-bar-display').as('bar');
        const getIJ = (i, j) => cy.get('@report').find(`[data-i=${i}][data-j=${j}]`).as(`${i} ${j}`);
        getIJ(-3, -2).click();
        cy.get('@bar').should('have.text', 'None');
        getIJ(-3, -2).type('{rightArrow}');
        cy.get('@bar').should('have.text', 'sum');
        getIJ(-3, 0).should('have.focus');
        getIJ(-3, 0).type('{downArrow}');
        cy.get('@bar').should('have.text', 'one');
        getIJ(-2, 0).type('{leftArrow}');
        cy.get('@bar').should('have.text', 'A');
        getIJ(-2, -2).should('have.focus');
        getIJ(-2, -2).type('{downArrow}');
        cy.get('@bar').should('have.text', 'B');
        getIJ(-1, -2).should('have.focus');
        getIJ(-1, -2).type('{downArrow}');
        cy.get('@bar').should('have.text', 'A');
        getIJ(0, -2).should('have.focus');
        getIJ(0, -2).type('{rightArrow}');
        cy.get('@bar').should('have.text', 'bar');
        getIJ(0, -1).type('{rightArrow}');
        getIJ(0, 0).should('have.focus');
        getIJ(0, 0).should('have.attr', 'data-role', 'dataframe-data');
    });
});
