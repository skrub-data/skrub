describe("Column filter search functionality", () => {
    describe("Text search input", () => {
        it("filters columns by plain text search", () => {
            cy.get("@report")
                .find('[data-test="column-filter-search"]')
                .as("searchInput");
            cy.get("@report").find('[data-test="n-columns-display"]').as("nColumns");

            // Initially all columns visible (8 columns in employee_salaries dataset)
            cy.get("@nColumns").should("have.text", "8");

            // Search for "date" - should match "date_first_hired"
            cy.get("@searchInput").type("date");
            cy.get("@nColumns").should("have.text", "1");

            cy.get("@report").find('[data-test="summaries-tab"]').click();
            cy.get("@report").find("#col_6").should("be.visible"); // date_first_hired
            cy.get("@report").find("#col_0").should("not.be.visible"); // gender

            // Clear and verify all columns are back
            cy.get("@searchInput").clear();
            cy.get("@nColumns").should("have.text", "8");
        });

        it("is case-insensitive", () => {
            cy.get("@report")
                .find('[data-test="column-filter-search"]')
                .as("searchInput");
            cy.get("@report").find('[data-test="n-columns-display"]').as("nColumns");

            cy.get("@searchInput").type("DEPARTMENT");
            cy.get("@nColumns").should("have.text", "2"); // department, department_name

            cy.get("@searchInput").clear().type("department");
            cy.get("@nColumns").should("have.text", "2");

            cy.get("@searchInput").clear().type("DePaRtMeNt");
            cy.get("@nColumns").should("have.text", "2");
        });

        it("handles partial matches", () => {
            cy.get("@report")
                .find('[data-test="column-filter-search"]')
                .as("searchInput");
            cy.get("@report").find('[data-test="n-columns-display"]').as("nColumns");

            // "date" should match "date_first_hired"
            cy.get("@searchInput").type("date");
            cy.get("@nColumns").should("have.text", "1");

            cy.get("@searchInput").clear().type("first");
            cy.get("@nColumns").should("have.text", "2");

            cy.get("@searchInput").clear().type("hired");
            cy.get("@nColumns").should("have.text", "2");
        });

        it("shows no columns when no matches found", () => {
            cy.get("@report")
                .find('[data-test="column-filter-search"]')
                .as("searchInput");
            cy.get("@report").find('[data-test="n-columns-display"]').as("nColumns");

            cy.get("@searchInput").type("nonexistent_column_xyz");
            cy.get("@nColumns").should("not.be.visible");

            cy.get("@report")
                .find('[data-test="sample-panel"]')
                .find('[data-test="show-all-columns-button"]')
                .should("be.visible");
        });

        it("trims whitespace from search text", () => {
            cy.get("@report")
                .find('[data-test="column-filter-search"]')
                .as("searchInput");
            cy.get("@report").find('[data-test="n-columns-display"]').as("nColumns");

            cy.get("@searchInput").type("    department    ");
            cy.get("@nColumns").should("have.text", "2"); // department, department_name
        });
    });

    describe("Regex toggle button", () => {
        it("enables and disables regex mode", () => {
            cy.get("@report")
                .find('[data-test="column-filter-regex"]')
                .as("regexButton");

            // Initially regex is disabled
            cy.get("@regexButton").should(
                "not.have.attr",
                "data-regex-enabled",
                "true"
            );

            // Enable regex
            cy.get("@regexButton").click();
            cy.get("@regexButton").should("have.attr", "data-regex-enabled", "true");

            // Disable regex
            cy.get("@regexButton").click();
            cy.get("@regexButton").should("have.attr", "data-regex-enabled", "false");
        });

        it("applies regex pattern when enabled", () => {
            cy.get("@report")
                .find('[data-test="column-filter-search"]')
                .as("searchInput");
            cy.get("@report")
                .find('[data-test="column-filter-regex"]')
                .as("regexButton");
            cy.get("@report").find('[data-test="n-columns-display"]').as("nColumns");

            // Enable regex mode
            cy.get("@regexButton").click();

            // Pattern to match columns ending with "e" - department_name, employee_position_title
            cy.get("@searchInput").type("e$");
            cy.get("@nColumns").should("have.text", "2");

            // Pattern to match columns starting with "d" - department, department_name, division, date_first_hired
            cy.get("@searchInput").clear().type("^d");
            cy.get("@nColumns").should("have.text", "4");
        });

        it("shows empty result for invalid regex", () => {
            cy.get("@report")
                .find('[data-test="column-filter-search"]')
                .as("searchInput");
            cy.get("@report")
                .find('[data-test="column-filter-regex"]')
                .as("regexButton");
            cy.get("@report").find('[data-test="n-columns-display"]').as("nColumns");

            cy.get("@regexButton").click();

            // Invalid regex pattern
            cy.get("@searchInput").type("[invalid(regex");
            cy.get("@nColumns").should("not.be.visible");

            cy.get("@report")
                .find('[data-test="sample-panel"]')
                .find('[data-test="show-all-columns-button"]')
                .should("be.visible");
        });

        it("switches between plain text and regex search", () => {
            cy.get("@report")
                .find('[data-test="column-filter-search"]')
                .as("searchInput");
            cy.get("@report")
                .find('[data-test="column-filter-regex"]')
                .as("regexButton");
            cy.get("@report").find('[data-test="n-columns-display"]').as("nColumns");

            // Plain text search for "department" - matches department, department_name
            cy.get("@searchInput").type("department");
            cy.get("@nColumns").should("have.text", "2");

            // Enable regex - pattern "^department$" matches exactly "department"
            cy.get("@regexButton").click();
            cy.get("@searchInput").clear().type("^department$");
            cy.get("@nColumns").should("have.text", "1");

            // Disable regex - literal search for "^department$" finds nothing
            cy.get("@regexButton").click();
            cy.get("@searchInput").clear().type("^department$");
            cy.get("@nColumns").should("not.be.visible");
        });
    });

    describe("Search combined with dropdown filter", () => {
        it("searches within the selected filter category", () => {
            cy.get("@report")
                .find('[data-test="column-filter-select"]')
                .as("filterSelect");
            cy.get("@report")
                .find('[data-test="column-filter-search"]')
                .as("searchInput");
            cy.get("@report").find('[data-test="n-columns-display"]').as("nColumns");

            // Select "Numeric" filter - only year_first_hired (col_7)
            cy.get("@filterSelect").select("Numeric");
            cy.get("@nColumns").should("have.text", "1");

            // Search for "year" within numeric columns - matches year_first_hired
            cy.get("@searchInput").type("year");
            cy.get("@nColumns").should("have.text", "1");

            // Search for "department" - not in numeric columns
            cy.get("@searchInput").clear().type("department");
            cy.get("@nColumns").should("not.be.visible");
        });

        it("maintains search text when changing dropdown filter", () => {
            cy.get("@report")
                .find('[data-test="column-filter-select"]')
                .as("filterSelect");
            cy.get("@report")
                .find('[data-test="column-filter-search"]')
                .as("searchInput");

            // Type search text
            cy.get("@searchInput").type("date");
            cy.get("@searchInput").should("have.value", "date");

            // Change dropdown filter
            cy.get("@filterSelect").select("Non-numeric");

            // Search text should still be there
            cy.get("@searchInput").should("have.value", "date");
        });

        it("updates filter display correctly", () => {
            cy.get('@report')
                .find('[data-test="column-filter-select"]')
                .as('filterSelect');
            cy.get("@report")
                .find('[data-test="column-filter-search"]')
                .as("searchInput");

            // Default value is disabled hidden hence null
            cy.get("@filterSelect").should("have.value", null);

            cy.get("@searchInput").type("department");
            cy.get("@filterSelect").should("have.value", null);

            cy.get("@searchInput").clear();
            cy.get("@filterSelect").select("Non-numeric");
            cy.get("@filterSelect")
                .find("option:selected")
                .invoke("text")
                .should((text) => {
                    expect(text.trim().toLowerCase()).to.contain("non-numeric");
                });

            cy.get("@searchInput").type("department");
            cy.get("@filterSelect")
                .find("option:selected")
                .invoke("text")
                .should((text) => {
                    expect(text.trim().toLowerCase()).to.contain("non-numeric");
                });
        });
    });

    describe("Reset functionality", () => {
        it("clears search input when reset button is clicked", () => {
            cy.get("@report")
                .find('[data-test="column-filter-search"]')
                .as("searchInput");
            cy.get("@report")
                .find('[data-test="show-all-columns-button"]')
                .as("resetButton");
            cy.get("@report")
                .find('[data-test="n-columns-display"]')
                .as("nColumns");

            // Enter search text
            cy.get("@searchInput").type("nonexistent_column_xyz");
            cy.get("@nColumns").should("have.text", "0");

            // Click reset
            cy.get("@resetButton").first().click({ force: true }); // in case button is hidden because of shadow-DOM
            cy.get("@searchInput").should("have.value", "");
            cy.get("@nColumns").should("have.text", "8");
        });

        it("disables regex when reset button is clicked", () => {
            cy.get("@report")
                .find('[data-test="column-filter-regex"]')
                .as("regexButton");
            cy.get("@report")
                .find('[data-test="show-all-columns-button"]')
                .as("resetButton");

            // Enable regex
            cy.get("@regexButton").click();
            cy.get("@regexButton").should("have.attr", "data-regex-enabled", "true");

            // Click reset
            cy.get("@resetButton").first().click({ force: true }); // in case button is hidden because of shadow-DOM
            cy.get("@regexButton").should("have.attr", "data-regex-enabled", "false");
        });

        it('resets dropdown to "All columns"', () => {
            cy.get("@report")
                .find('[data-test="column-filter-select"]')
                .as("filterSelect");
            cy.get("@report")
                .find('[data-test="show-all-columns-button"]')
                .as("resetButton");
            cy.get("@report")
                .find('[data-test="n-columns-display"]')
                .as("nColumns");

            // Change filter
            cy.get("@filterSelect").select("Numeric");
            cy.get("@nColumns").should("have.text", "1");

            // Click reset
            cy.get("@resetButton").first().click({ force: true }); // in case button is hidden because of shadow-DOM
            cy.get("@filterSelect").should("have.value", null); // "All columns"
            cy.get("@nColumns").should("have.text", "8");
        });

        it("resets all filter controls together", () => {
            cy.get("@report")
                .find('[data-test="column-filter-select"]')
                .as("filterSelect");
            cy.get("@report")
                .find('[data-test="column-filter-search"]')
                .as("searchInput");
            cy.get("@report")
                .find('[data-test="column-filter-regex"]')
                .as("regexButton");
            cy.get("@report")
                .find('[data-test="show-all-columns-button"]')
                .as("resetButton");
            cy.get("@report")
                .find('[data-test="n-columns-display"]')
                .as("nColumns");

            // Apply multiple filters
            cy.get("@filterSelect").select("Non-numeric");
            cy.get("@regexButton").click();
            cy.get("@searchInput").type("^department");
            cy.get("@nColumns").should("have.text", "2"); // department, department_name

            // Reset everything
            cy.get("@resetButton").first().click({ force: true }); // in case button is hidden because of shadow-DOM
            cy.get("@filterSelect").should("have.value", null); // "All columns"
            cy.get("@searchInput").should("have.value", "");
            cy.get("@regexButton").should("have.attr", "data-regex-enabled", "false");
            cy.get("@nColumns").should("have.text", "8");
        });
    });

    describe("Match count display", () => {
        it("shows correct count for search results", () => {
            cy.get("@report")
                .find('[data-test="column-filter-search"]')
                .as("searchInput");
            cy.get("@report")
                .find('[data-test="n-columns-display"]')
                .as("matchCount");

            cy.get("@matchCount").should("have.text", "8");

            cy.get("@searchInput").type("date");
            cy.get("@matchCount").should("have.text", "1");
        });

        it("shows 0 when no matches", () => {
            cy.get("@report")
                .find('[data-test="column-filter-search"]')
                .as("searchInput");
            cy.get("@report")
                .find('[data-test="n-columns-display"]')
                .as("matchCount");

            cy.get("@searchInput").type("xyz_nonexistent");
            cy.get("@matchCount").should("have.text", "0");
        });

        it("updates when switching between dropdown filters", () => {
            cy.get("@report")
                .find('[data-test="column-filter-select"]')
                .as("filterSelect");
            cy.get("@report")
                .find('[data-test="n-columns-display"]')
                .as("matchCount");

            cy.get("@matchCount").should("have.text", "8");

            cy.get("@filterSelect").select("Numeric");
            cy.get("@matchCount").should("have.text", "1");

            cy.get("@filterSelect").select("Non-numeric");
            cy.get("@matchCount").should("have.text", "7");
        });
    });

    describe("Edge cases", () => {
        it("handles empty search string", () => {
            cy.get("@report")
                .find('[data-test="column-filter-search"]')
                .as("searchInput");
            cy.get("@report").find('[data-test="n-columns-display"]').as("nColumns");

            cy.get("@searchInput").type("     ");
            cy.get("@nColumns").should("have.text", "8");
        });

        it("handles regex special characters in plain text mode", () => {
            cy.get("@report")
                .find('[data-test="column-filter-search"]')
                .as("searchInput");
            cy.get("@report").find('[data-test="n-columns-display"]').as("nColumns");

            // These should be treated as literal characters, not regex
            cy.get("@searchInput").type(".*");
            cy.get("@nColumns").should("have.text", "0");
        });

        it("persists filter state across tab switches", () => {
            cy.get("@report")
                .find('[data-test="column-filter-search"]')
                .as("searchInput");
            cy.get("@report").find('[data-test="n-columns-display"]').as("nColumns");

            // Apply search filter for "year" - matches year_first_hired (col_7)
            cy.get("@searchInput").type("year");
            cy.get("@nColumns").should("have.text", "1");

            // Switch to summaries tab
            cy.get("@report").find('[data-test="summaries-tab"]').click();
            cy.get("@report").find("#col_7").should("be.visible");

            // Switch back to sample tab
            cy.get("@report").find('[data-test="sample-tab"]').click();
            cy.get("@nColumns").should("have.text", "1");
        });

        it("handles multiple sequential searches", () => {
            cy.get("@report")
                .find('[data-test="column-filter-search"]')
                .as("searchInput");
            cy.get("@report").find('[data-test="n-columns-display"]').as("nColumns");

            cy.get("@searchInput").type("department");
            cy.get("@nColumns").should("have.text", "2"); // department, department_name

            cy.get("@searchInput").clear().type("date");
            cy.get("@nColumns").should("have.text", "1"); // date_first_hired

            cy.get("@searchInput").clear().type("year");
            cy.get("@nColumns").should("have.text", "1"); // year_first_hired

            cy.get("@searchInput").clear();
            cy.get("@nColumns").should("have.text", "8");
        });

        it("works correctly when toggling regex mid-search", () => {
            cy.get("@report")
                .find('[data-test="column-filter-search"]')
                .as("searchInput");
            cy.get("@report")
                .find('[data-test="column-filter-regex"]')
                .as("regexButton");
            cy.get("@report").find('[data-test="n-columns-display"]').as("nColumns");

            // Type a pattern that works as both literal and regex
            cy.get("@searchInput").type("department");
            cy.get("@nColumns").should("have.text", "2"); // department, department_name

            // Enable regex - should still work
            cy.get("@regexButton").click();
            cy.get("@nColumns").should("have.text", "2");

            // Now add regex syntax for alternation - department|gender
            cy.get("@searchInput").clear().type("department|gender");
            cy.get("@nColumns").should("have.text", "3"); // department, department_name, gender

            // Disable regex - should not match anything (literal "department|gender")
            cy.get("@regexButton").click();
            cy.get("@nColumns").should("not.be.visible");
        });
    });
});
