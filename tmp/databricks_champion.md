%md-sandbox
### B5. Create the Policy in the UI

<div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 14px 18px; border-radius: 4px; margin: 16px 0;">
<strong style="display:block; color:#0d47a1; margin-bottom:6px; font-size: 1.1em;">Step 2: Navigate to your schema</strong>
<div style="color:#333;">Open <strong>Catalog</strong> from the left sidebar. Navigate to <strong>dbacademy</strong> > <strong>your_schema</strong> (e.g., <code>labuser15359237_1780317248</code>). Click the <strong>Policies</strong> tab, then click <strong>New Policy</strong>.</div>
</div>

<div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 14px 18px; border-radius: 4px; margin: 16px 0;">
<strong style="display:block; color:#0d47a1; margin-bottom:6px; font-size: 1.1em;">Step 3: Configure the policy basics</strong>
<div style="color:#333;">
<ol>
<li><strong>Name:</strong> <code>address_policy</code></li>
<li><strong>Description:</strong> Mask the address column in all tables within this schema.</li>
<li><strong>Applied to:</strong> All account users</li>
<li><strong>Except for:</strong> <code>metastore_admins</code></li>
<li><strong>Scope:</strong> <code>dbacademy.your_schema.All tables</code> (select your schema from the dropdown)</li>
</ol>
</div>
</div>

<div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 14px 18px; border-radius: 4px; margin: 16px 0;">
<strong style="display:block; color:#0d47a1; margin-bottom:6px; font-size: 1.1em;">Step 4: Set the policy type and conditions</strong>
<div style="color:#333;">
<ol>
<li><strong>Policy type:</strong> Select <strong>Column mask</strong> (replaces column values with masked versions).</li>
<li><strong>Condition:</strong> Leave as <strong>No condition</strong> (all tables in scope).</li>
<li><strong>Column conditions:</strong> Choose one of these approaches:
  <ul>
    <li><strong>Option A (Tag matching):</strong> Select <strong>Columns matching any of these tags</strong>. Add the tag <code>pii : address</code>. The system generates: <code>hasTagValue('pii','address')</code>.</li>
    <li><strong>Option B (Custom expression):</strong> Select <strong>Columns matching a custom expression</strong>. Type: <code>hasTagValue('pii','address')</code></li>
  </ul>
</li>
</ol>
</div>
</div>

<div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 14px 18px; border-radius: 4px; margin: 16px 0;">
<strong style="display:block; color:#0d47a1; margin-bottom:6px; font-size: 1.1em;">Step 5: Create the masking function</strong>
<div style="color:#333;">
<ol>
<li>In the <strong>Masking function</strong> section, toggle to <strong>Create</strong> (not "Select Existing").</li>
<li>Make sure the <strong>Shared Warehouse</strong> is selected and running in the top-right dropdown.</li>
<li>Click the <strong>AI assistant icon</strong> (sparkle icon on the right).</li>
<li>In the prompt field, type: <code>show first 3 characters</code></li>
<li>Click <strong>Generate</strong>. The assistant generates a function that preserves the first 3 characters and masks the rest.</li>
<li>Click <strong>Accept</strong> to apply the generated function.</li>
<li>Click <strong>Check syntax</strong> to validate. (If the button is greyed out, start the Shared Warehouse first.)</li>
<li>Expand <strong>Test function</strong>, enter a test value (e.g., <code>testingemail@email.com</code>), and click <strong>Run test</strong>. Verify the output shows the first 3 characters followed by asterisks (e.g., <code>tes*******************</code>).</li>
<li>Click <strong>Create policy</strong>.</li>
</ol>
</div>
</div>

<div style="border-left: 4px solid #ff9800; background: #fff3e0; padding: 14px 18px; border-radius: 4px; margin: 16px 0;">
<strong style="display:block; color:#e65100; margin-bottom:6px; font-size: 1.1em;">Important</strong>
<div style="color:#333;">ABAC policies apply to SDP pipeline outputs (streaming tables and materialized views). If you plan to rerun the pipeline, either remove the policy first or add your <code>labuser@vocareum.com</code> to the exemption list to avoid refresh failures.</div>
</div>

<div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 14px 18px; border-radius: 4px; margin: 16px 0;">
<strong style="display:block; color:#0d47a1; margin-bottom:6px; font-size: 1.1em;">Step 6: Verify the policy</strong>
<div style="color:#333;">Run the code cells below to query tables with an <code>address</code> column. The address values should now show only the first 3 characters with the rest masked.</div>
</div>